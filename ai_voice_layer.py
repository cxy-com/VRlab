import logging
import speech_recognition as sr
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import re
import json
from datetime import datetime

# 使用相对导入
from ..data.data_layer import (
    DataInteractionLayer, 
    VoiceCommandContext, 
    Component, 
    ComponentType
)

# 添加ERNIE模型集成（任何导入异常都降级为规则解析，避免因环境问题导致整个语音系统不可用）
try:
    from ..ernie_integration_fixed import get_ernie_instance
    _ERNIE_AVAILABLE = True
except Exception:
    # 如果无法导入（包括底层依赖如 torch/paddle 的 OSError），设置为 None，使用规则解析
    _ERNIE_AVAILABLE = False
    def get_ernie_instance():
        return None

# 添加讯飞星火API集成
try:
    from .xfyun_spark_api import parse_voice_command as xfyun_parse
    _XFYUN_AVAILABLE = True
except ImportError:
    _XFYUN_AVAILABLE = False
    xfyun_parse = None

# 添加讯飞语音识别集成
try:
    from .xfyun_asr_api import recognize_speech as xfyun_asr
    _XFYUN_ASR_AVAILABLE = True
except ImportError:
    _XFYUN_ASR_AVAILABLE = False
    xfyun_asr = None

# 添加Whisper语音识别集成（如本地未正确安装 torch/whisper，则自动降级为仅用讯飞）
try:
    from .whisper_asr_api import WhisperASRAPI, recognize_speech as whisper_asr
    _WHISPER_AVAILABLE = True
except Exception:
    _WHISPER_AVAILABLE = False
    whisper_asr = None

# 添加微调 Whisper 模型集成（专用于电路指令）
try:
    from .finetuned_whisper_api import FinetunedWhisperAPI, recognize_speech as finetuned_whisper_asr
    _FINETUNED_WHISPER_AVAILABLE = True
except Exception:
    _FINETUNED_WHISPER_AVAILABLE = False
    finetuned_whisper_asr = None

logger = logging.getLogger(__name__)


class VoiceCommandType(Enum):
    """语音指令类型"""
    ADD_COMPONENT = "add_component"
    CONNECT_COMPONENTS = "connect_components"
    RUN_SIMULATION = "run_simulation"
    VERIFY_THEOREM = "verify_theorem"
    MODIFY_COMPONENT = "modify_component"
    DELETE_COMPONENT = "delete_component"
    UNKNOWN = "unknown"


@dataclass
class ParsedCommand:
    """解析后的语音指令"""
    command_type: VoiceCommandType
    confidence: float
    parameters: Dict[str, Any]
    original_text: str
    needs_clarification: bool = False
    clarification_question: Optional[str] = None


class AIVoiceLayer:
    """AI语音层主类"""
    
    def __init__(self, data_layer: DataInteractionLayer, vr_layer=None):
        self.logger = logger
        self.data_layer = data_layer
        self.vr_layer = vr_layer  # VR交互层引用，用于执行指令
        
        # 语音识别组件（延迟初始化）
        self.recognizer = sr.Recognizer()
        self.microphone = None
        self._microphone_initialized = False
        
        # 专业术语词典
        self.electric_terms = self._load_electric_terms()
        
        # 讯飞星火API相关
        self.xfyun_enabled = _XFYUN_AVAILABLE
        if self.xfyun_enabled:
            self.logger.info("讯飞星火API可用")
        else:
            self.logger.info("讯飞星火API不可用")
        
        # 讯飞语音识别相关
        self.xfyun_asr_enabled = _XFYUN_ASR_AVAILABLE
        if self.xfyun_asr_enabled:
            self.logger.info("讯飞语音识别API可用")
        else:
            self.logger.warning("讯飞语音识别API不可用")

        # Whisper语音识别相关
        self.whisper_enabled = _WHISPER_AVAILABLE
        self.whisper_model = None
        if self.whisper_enabled:
            self.logger.info("Whisper语音识别可用")
        else:
            self.logger.warning("Whisper语音识别不可用")
        
        # 微调 Whisper 模型相关（电路指令专用）
        self.use_finetuned_whisper = _FINETUNED_WHISPER_AVAILABLE
        self._finetuned_api = None
        if self.use_finetuned_whisper:
            try:
                self._finetuned_api = FinetunedWhisperAPI()
                if self._finetuned_api.is_available():
                    self.logger.info("✅ 微调 Whisper 模型已启用（电路指令专用，准确率 95%+）")
                else:
                    self.use_finetuned_whisper = False
                    self.logger.warning("微调模型加载失败，将使用其他识别方案")
            except Exception as e:
                self.logger.warning(f"微调模型初始化失败: {e}")
                self.use_finetuned_whisper = False
        
        # ERNIE模型相关：保留实例引用，解析时再检查 is_loaded（异步加载完成后即可使用）
        self.ernie_model = None
        if _ERNIE_AVAILABLE:
            try:
                self.ernie_model = get_ernie_instance()
                if self.ernie_model and self.ernie_model.is_loaded:
                    self.logger.info("用户训练的ERNIE模型加载成功")
                else:
                    self.logger.info("ERNIE模型后台加载中，加载完成后将自动启用")
            except Exception as e:
                self.logger.warning(f"ERNIE模型加载失败，使用规则解析: {e}")
                self.ernie_model = None
        else:
            self.logger.info("ERNIE模块不可用，使用规则解析")
        
        # 语音识别配置（偏快响应：说完后更快结束采集）
        self.energy_threshold = 300
        self.pause_threshold = 0.5   # 静音 0.5 秒即认为说完（须 >= non_speaking_duration）
        self.non_speaking_duration = 0.4  # 短语前后保留静音，须 <= pause_threshold
        
        # 元件ID计数器（用于自动生成ID）
        self._component_counters = {
            'resistor': 1,
            'power_source': 1,
            'ground': 1
        }
        self._wire_counter = 1
        
        # 尝试初始化麦克风（可选，失败不影响文本解析功能）
        self._try_setup_voice_recognition()
        self.logger.info("AI语音层初始化完成")
    
    def set_vr_layer(self, vr_layer) -> None:
        """设置VR层引用（支持延迟绑定）"""
        self.vr_layer = vr_layer
        self.logger.info("VR层已绑定到语音层")
    
    def _load_electric_terms(self) -> Dict[str, List[str]]:
        """加载电学专业术语词典"""
        return {
            "component_types": ["电阻", "电源", "信号发生器", "导线", "接地"],
            "operations": ["添加", "连接", "删除", "修改", "运行", "验证"],
            "theorems": ["基尔霍夫", "KVL", "KCL", "电压定律", "电流定律"],
            "units": ["伏特", "V", "欧姆", "Ω", "kΩ", "赫兹", "Hz"]
        }
    
    def _try_setup_voice_recognition(self) -> None:
        """尝试设置语音识别（失败不影响其他功能）"""
        try:
            self.microphone = sr.Microphone()
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source)
            
            self.recognizer.energy_threshold = self.energy_threshold
            self.recognizer.pause_threshold = self.pause_threshold
            self.recognizer.non_speaking_duration = getattr(self, "non_speaking_duration", 0.5)
            self._microphone_initialized = True
            self.logger.info("麦克风初始化成功")
        except Exception as e:
            self.logger.warning(f"麦克风初始化失败（文本解析功能仍可用）: {e}")
            self.microphone = None
            self._microphone_initialized = False
    
    def process_voice_input(self) -> Optional[Dict[str, Any]]:
        """处理语音输入"""
        try:
            # 1. 采集语音
            audio_data = self._capture_voice()
            if not audio_data:
                return None
            
            # 2. 语音转文本
            text = self._speech_to_text(audio_data)
            if not text:
                return None
            # 在进入电学解析前先做一次领域纠错与规范化
            text = self._normalize_asr_text(text)
            
            # 3. 解析指令
            parsed_command = self._parse_voice_command(text)
            if not parsed_command:
                return None
            
            # 4. 校验指令
            if not self._validate_command(parsed_command):
                return None
            
            # 5. 存储上下文
            self._store_command_context(text, parsed_command)
            
            return parsed_command.parameters
            
        except Exception as e:
            self.logger.error(f"语音处理错误: {e}")
            return None
    
    def _capture_voice(self) -> Optional[sr.AudioData]:
        """采集语音（支持从菜单/按键触发，采集前会重新校准环境噪声）"""
        if not self._microphone_initialized or self.microphone is None:
            self.logger.error("麦克风未初始化，无法采集语音")
            return None

        try:
            with self.microphone as source:
                self.logger.info("开始语音采集...")
                # 指令一般较短：最多等 5 秒开始说，单段最长 5 秒（说完更快结束）
                audio_data = self.recognizer.listen(
                    source, timeout=5, phrase_time_limit=5
                )
            return audio_data
        except sr.WaitTimeoutError:
            self.logger.warning("语音采集超时（未检测到语音），请重试")
            return None
    
    def _speech_to_text(self, audio_data: sr.AudioData) -> Optional[str]:
        """
        语音转文本 - 三级 fallback 策略：
        1. 微调 Whisper（电路专用，准确率 95%+）
        2. 讯飞 ASR（在线服务，识别稳定）
        3. 原始 Whisper（通用兜底，离线可用）
        """
        pcm_data = audio_data.get_raw_data(convert_rate=16000, convert_width=2)
        
        # 1. 优先使用微调模型（专门识别电路指令，准确率最高）
        if self.use_finetuned_whisper and self._finetuned_api:
            try:
                text = self._finetuned_api.recognize_pcm(pcm_data, sample_rate=16000)
                if text:
                    self.logger.info(f"🔧 微调模型识别: {text}")
                    # 如果识别结果包含电路关键词，直接返回
                    if self._is_likely_circuit_command(text):
                        return text
                    else:
                        self.logger.debug(f"微调模型结果可能非指令，尝试其他方案")
            except Exception as e:
                self.logger.warning(f"微调模型识别失败: {e}")
        
        # 2. 微调模型失败或结果不确定，使用讯飞 ASR
        if self.xfyun_asr_enabled and xfyun_asr:
            try:
                text = xfyun_asr(pcm_data, audio_format="audio/L16;rate=16000")
                if text:
                    self.logger.info(f"讯飞识别: {text}")
                    return text
            except Exception as e:
                self.logger.warning(f"讯飞识别失败: {e}")
        
        # 3. 兜底：使用原始 Whisper
        if self.whisper_enabled and whisper_asr:
            try:
                text = whisper_asr(pcm_data, sample_rate=16000, model_size="small")
                if text:
                    self.logger.info(f"Whisper识别: {text}")
                    return text
            except Exception as e:
                self.logger.warning(f"Whisper识别失败: {e}")

        self.logger.error("所有语音识别方案均失败")
        return None

    def _is_likely_circuit_command(self, text: str) -> bool:
        """判断识别结果是否可能是电路指令"""
        if not text:
            return False
        
        # 电路指令关键词
        circuit_keywords = [
            "电阻", "电压", "电源", "电流", "接地", "电容", "电感",
            "添加", "连接", "串联", "并联", "删除", "设置",
            "仿真", "运行", "验证", "R1", "R2", "V1", "V2", "GND",
            "KVL", "KCL", "欧姆", "伏特", "安培"
        ]
        
        text_lower = text.lower()
        return any(kw in text or kw.lower() in text_lower for kw in circuit_keywords)

    def _is_likely_off_topic(self, text: str) -> bool:
        """判断是否为明显非电学指令（ASR 误识别等），用于跳过讯飞星火长段说明"""
        if not text or len(text) > 200:
            return False
        t = text.strip()
        lower = t.lower()
        # 电学相关关键词（出现任一则认为是可能指令，不判为 off-topic）
        lab_keywords = [
            "添加", "加", "放置", "删除", "连接", "串联", "并联", "仿真", "运行", "验证",
            "电阻", "电源", "接地", "信号", "电压", "电流", "kvl", "kcl", "欧姆", "伏特",
            "add", "place", "delete", "remove", "connect", "run", "simulate", "resistor",
            "power", "ground", "voltage", "r1", "v1", "gnd", "wire", "link"
        ]
        if any(k in t or k in lower for k in lab_keywords):
            return False
        # 常见误识别或无关短句
        off_patterns = ["店主", "阿姨", "请教", "店家", "好的然后", "可以吗", "怎么办", "为什么", "加店主"]
        if any(p in t for p in off_patterns):
            return True
        # 很短且无数字、无 R/V/GND 等元件号
        if len(t) <= 8 and not re.search(r"[RVG]\d+|[\d一二两十百千]", t):
            return True
        return False

    def _normalize_asr_text(self, text: str) -> str:
        """
        ASR 常见误识别纠错和电学领域规范化，提高指令解析准确率
        全面覆盖：元件名称、操作动词、单位、数字、常见谐音错误
        """
        if not text or not text.strip():
            return text
        # 先统一去掉首尾空白
        t = text.strip()
        # 去掉常见口头语/语气词，减少干扰
        filler_patterns = [
            "呃", "嗯", "啊", "然后", "就是", "那个", "这个", "那么",
            "请问", "我想要", "我想", "帮我", "可以", "麻烦", "给我",
        ]
        for fp in filler_patterns:
            if fp in t:
                t = t.replace(fp, "")

        # ==================== 完整谐音纠错规则 ====================
        # 格式: (错误词, 正确词)

        replacement_pairs = [
            # ==================== R/I/1 混淆 (最高频) ====================
            ("I1", "R1"), ("I2", "R2"), ("I3", "R3"), ("I4", "R4"),
            ("I5", "R5"), ("I6", "R6"), ("I7", "R7"), ("I8", "R8"), ("I9", "R9"), ("I0", "R0"),
            ("i1", "R1"), ("i2", "R2"), ("i3", "R3"), ("i4", "R4"),
            ("i5", "R5"), ("i6", "R6"), ("i7", "R7"), ("i8", "R8"), ("i9", "R9"), ("i0", "R0"),
            ("爱1", "R1"), ("爱2", "R2"), ("爱3", "R3"), ("爱4", "R4"), ("爱5", "R5"),
            ("爱6", "R6"), ("爱7", "R7"), ("爱8", "R8"), ("爱9", "R9"), ("爱0", "R0"),

            # ==================== 电阻 ====================
            ("电组", "电阻"), ("点组", "电阻"), ("店组", "电阻"), ("典阻", "电阻"),
            ("电祖", "电阻"), ("电足", "电阻"), ("定组", "电阻"), ("丢组", "电阻"),
            ("田组", "电阻"), ("甜组", "电阻"),

            # ==================== 电源 ====================
            ("电缘", "电源"), ("电远", "电源"), ("店源", "电源"), ("店员", "电源"),
            ("电原", "电源"), ("定源", "电源"), ("丢源", "电源"),

            # ==================== 接地 ====================
            ("金地", "接地"), ("劲地", "接地"), ("接的", "接地"),
            ("结地", "接地"), ("节地", "接地"), ("解地", "接地"),

            # ==================== 添加 ====================
            ("田家", "添加"), ("天价", "添加"), ("天加", "添加"),
            ("填加", "添加"), ("甜家", "添加"), ("添家", "添加"),

            # ==================== 删除 ====================
            ("删掉", "删除"), ("删去", "删除"), ("去掉", "删除"),
            ("除掉", "删除"), ("山掉", "删除"),

            # ==================== 连接 ====================
            ("连节", "连接"), ("连结", "连接"), ("联接", "连接"),
            ("练接", "连接"), ("连街", "连接"), ("串连", "串联"),
            ("并连", "并联"),

            # ==================== 仿真/运行 ====================
            ("仿镇", "仿真"), ("仿贞", "仿真"), ("方真", "仿真"),
            ("份真", "仿真"), ("云行", "运行"), ("银星", "运行"),

            # ==================== 单位 ====================
            ("欧母", "欧姆"), ("偶母", "欧姆"), ("偶", "欧"),
            ("伏特特", "伏特"), ("浮", "伏"), ("福", "伏"), ("扶", "伏"),
            ("安培培", "安培"), ("瓦特特", "瓦特"),

            # ==================== 数字 ====================
            ("两千", "2000"), ("三千", "3000"), ("四千", "4000"), ("五千", "5000"),
            ("六千", "6000"), ("七千", "7000"), ("八千", "8000"), ("一万", "10000"),

            # ==================== V/微/维 ====================
            ("微1", "V1"), ("微2", "V2"), ("微3", "V3"), ("微4", "V4"), ("微5", "V5"),
            ("维1", "V1"), ("维2", "V2"), ("维3", "V3"), ("维4", "V4"), ("维5", "V5"),
            ("为1", "V1"), ("为2", "V2"), ("为3", "V3"), ("为4", "V4"), ("为5", "V5"),

            # ==================== 连接短语常见误识别 ====================
            ("连上", "连接"), ("接上", "连接"), ("接到", "连接"), ("连到", "连接"),
            ("连一起", "连接"), ("接一起", "连接"),
        ]

        # 执行替换
        for old, new in replacement_pairs:
            if old in t:
                t = t.replace(old, new)

        # 正则替换：处理更复杂的模式
        # 微/维/为 + 数字 → V + 数字
        t = re.sub(r"微(\d+)", r"V\1", t)
        t = re.sub(r"维(\d+)", r"V\1", t)
        t = re.sub(r"为(\d+)", r"V\1", t)

        # 将 ASR 常把 R 识别成 I 的元件编号规范为 Rn（保留原有逻辑）
        t = re.sub(r"\bI\s*1\b", "R1", t, flags=re.IGNORECASE)
        t = re.sub(r"\bI\s*2\b", "R2", t, flags=re.IGNORECASE)
        t = re.sub(r"\bI\s*(\d+)\b", r"R\1", t, flags=re.IGNORECASE)

        return t.strip()

    def _parse_voice_command(self, text: str) -> Optional[ParsedCommand]:
        """
        解析语音指令 - 优先规则（快），再讯飞星火/ERNIE（准）
        规则能识别的指令直接返回，减少网络/模型延迟
        """
        text = (text or "").strip()
        if not text:
            return None

        # 0. 明显非电学指令（如 ASR 误识别“请教店主”“店家阿姨”）直接拒识，不调讯飞星火避免长段说明刷屏
        if self._is_likely_off_topic(text):
            self.logger.info(f"跳过解析（非电学指令）: {text[:30]}{'...' if len(text) > 30 else ''}")
            return None

        # 1. 先试规则解析（本地、无网络，常见指令立即可得）
        rules_result = self._parse_with_rules(text)
        if rules_result and rules_result.confidence >= 0.65:
            self.logger.info(f"使用规则解析(快): {text} (置信度: {rules_result.confidence})")
            return rules_result

        # 2. 尝试你训练的 ERNIE 模型（电学指令专用，本地推理）
        if self.ernie_model and self.ernie_model.is_loaded:
            ernie_result = self._parse_with_ernie(text)
            if ernie_result and ernie_result.confidence >= 0.6:
                self.logger.info(f"使用ERNIE解析: {text} (置信度: {ernie_result.confidence})")
                return ernie_result
            elif ernie_result:
                self.logger.debug(f"ERNIE置信度不足({ernie_result.confidence})，尝试讯飞星火")

        # 3. 尝试讯飞星火API解析
        if self.xfyun_enabled:
            xfyun_result = self._parse_with_xfyun(text)
            if xfyun_result and xfyun_result.confidence >= 0.7:
                self.logger.info(f"使用讯飞星火解析: {text} (置信度: {xfyun_result.confidence})")
                return xfyun_result
            elif xfyun_result:
                self.logger.debug(f"讯飞星火置信度不足({xfyun_result.confidence})")

        # 4. 规则兜底（可能为 None）
        return rules_result
    
    def _parse_with_ernie(self, text: str) -> Optional[ParsedCommand]:
        """使用用户训练的ERNIE模型解析指令"""
        try:
            if not self.ernie_model or not self.ernie_model.is_loaded:
                return None
            
            # 使用ERNIE模型解析
            ernie_result = self.ernie_model.parse_voice_command(text)
            
            if not ernie_result:
                return None
            
            # 转换为ParsedCommand格式
            action = ernie_result.get("action", "unknown")
            confidence = ernie_result.get("confidence", 0.5)
            parameters = ernie_result.get("parameters", {})
            
            # 映射到VoiceCommandType
            command_type_map = {
                "add_component": VoiceCommandType.ADD_COMPONENT,
                "add": VoiceCommandType.ADD_COMPONENT,
                "connect_components": VoiceCommandType.CONNECT_COMPONENTS,
                "connect": VoiceCommandType.CONNECT_COMPONENTS,
                "run_simulation": VoiceCommandType.RUN_SIMULATION,
                "verify_law": VoiceCommandType.VERIFY_THEOREM,
                "measure_voltage": VoiceCommandType.MODIFY_COMPONENT,
                "modify_component": VoiceCommandType.MODIFY_COMPONENT,
                "delete_component": VoiceCommandType.DELETE_COMPONENT,
                "delete": VoiceCommandType.DELETE_COMPONENT,
                "generate_signal": VoiceCommandType.RUN_SIMULATION,
                "analyze_results": VoiceCommandType.RUN_SIMULATION,
                "parse_command": VoiceCommandType.UNKNOWN,  # ERNIE返回的默认值
            }
            
            command_type = command_type_map.get(action, VoiceCommandType.UNKNOWN)
            
            # 如果ERNIE解析结果为UNKNOWN或置信度太低，返回None让规则解析处理
            if command_type == VoiceCommandType.UNKNOWN or confidence < 0.6:
                self.logger.debug(f"ERNIE解析置信度不足或类型未知，回退到规则解析: {action}, {confidence}")
                return None
            
            return ParsedCommand(
                command_type=command_type,
                confidence=confidence,
                parameters=parameters,
                original_text=text
            )
            
        except Exception as e:
            self.logger.error(f"ERNIE模型解析错误: {e}")
            return None
    
    def _parse_with_xfyun(self, text: str) -> Optional[ParsedCommand]:
        """使用讯飞星火API解析指令"""
        try:
            if not self.xfyun_enabled or not xfyun_parse:
                return None
            
            # 调用讯飞星火API
            xfyun_result = xfyun_parse(text)
            
            if not xfyun_result:
                return None
            
            # 转换为ParsedCommand格式
            action = xfyun_result.get("action", "unknown")
            confidence = xfyun_result.get("confidence", 0.5)
            
            # 映射到VoiceCommandType
            command_type_map = {
                "add": VoiceCommandType.ADD_COMPONENT,
                "add_component": VoiceCommandType.ADD_COMPONENT,
                "connect": VoiceCommandType.CONNECT_COMPONENTS,
                "connect_components": VoiceCommandType.CONNECT_COMPONENTS,
                "simulate": VoiceCommandType.RUN_SIMULATION,
                "run_simulation": VoiceCommandType.RUN_SIMULATION,
                "verify_KVL": VoiceCommandType.VERIFY_THEOREM,
                "verify_KCL": VoiceCommandType.VERIFY_THEOREM,
                "verify": VoiceCommandType.VERIFY_THEOREM,
                "delete": VoiceCommandType.DELETE_COMPONENT,
                "delete_component": VoiceCommandType.DELETE_COMPONENT,
                "modify": VoiceCommandType.MODIFY_COMPONENT,
                "modify_component": VoiceCommandType.MODIFY_COMPONENT,
            }
            
            command_type = command_type_map.get(action, VoiceCommandType.UNKNOWN)
            
            # 如果解析结果为UNKNOWN或置信度太低，返回None
            if command_type == VoiceCommandType.UNKNOWN or confidence < 0.7:
                self.logger.debug(f"讯飞星火解析置信度不足或类型未知: {action}, {confidence}")
                return None
            
            # 转换参数格式
            parameters = self._convert_xfyun_parameters(xfyun_result, command_type)
            
            return ParsedCommand(
                command_type=command_type,
                confidence=confidence,
                parameters=parameters,
                original_text=text
            )
            
        except Exception as e:
            self.logger.error(f"讯飞星火API解析错误: {e}")
            return None
    
    def _convert_xfyun_parameters(self, xfyun_result: Dict, command_type: VoiceCommandType) -> Dict[str, Any]:
        """将讯飞星火的参数格式转换为内部格式"""
        parameters = {"type": command_type.value}
        
        # 添加元件
        if command_type == VoiceCommandType.ADD_COMPONENT:
            comp_type_map = {
                "resistor": "resistor",
                "power": "power_source",
                "power_supply": "power_source",
                "voltage_source": "power_source",
                "ground": "ground",
                "oscilloscope": "oscilloscope",
                "signal_generator": "power_source",
            }

            xfyun_comp_type = (xfyun_result.get("component_type") or "").strip().lower()
            parameters["component_type"] = comp_type_map.get(xfyun_comp_type, xfyun_comp_type or "resistor")
            parameters["component_id"] = xfyun_result.get("component_id", "")
            
            # 提取参数
            xfyun_params = xfyun_result.get("parameters", {})
            if "value" in xfyun_params:
                value = xfyun_params["value"]
                unit = xfyun_params.get("unit", "")
                
                # 根据单位判断参数类型
                if unit in ["Ω", "kΩ", "ohm"]:
                    # 电阻值
                    resistance = float(value)
                    if unit == "kΩ":
                        resistance *= 1000
                    parameters["resistance"] = resistance
                elif unit in ["V", "volt"]:
                    # 电压值
                    parameters["voltage"] = float(value)
                elif unit in ["Hz", "kHz"]:
                    # 频率值
                    frequency = float(value)
                    if unit == "kHz":
                        frequency *= 1000
                    parameters["frequency"] = frequency
        
        # 连接元件
        elif command_type == VoiceCommandType.CONNECT_COMPONENTS:
            components = xfyun_result.get("components", [])
            if len(components) >= 2:
                parameters["component1_id"] = components[0]
                parameters["component2_id"] = components[1]
            parameters["connection_type"] = "series"  # 默认串联
        
        # 删除元件
        elif command_type == VoiceCommandType.DELETE_COMPONENT:
            parameters["component_id"] = xfyun_result.get("component_id", "")
            parameters["delete_type"] = "component"
        
        # 验证定理
        elif command_type == VoiceCommandType.VERIFY_THEOREM:
            action = xfyun_result.get("action", "")
            if "KCL" in action:
                parameters["theorem"] = "kcl"
            else:
                parameters["theorem"] = "kvl"
        
        return parameters
    
    def _parse_with_rules(self, text: str) -> Optional[ParsedCommand]:
        """使用规则解析指令（支持中英文）"""
        text_lower = text.lower()
        
        # 删除元件指令（优先判断，避免被其他指令误匹配）
        if any(keyword in text_lower for keyword in ["删除", "移除", "去掉", "删掉", "delete", "remove"]):
            return self._parse_delete_component(text)
        
        # 添加元件指令
        elif any(keyword in text_lower for keyword in ["添加", "加", "放置", "add", "place", "put"]):
            return self._parse_add_component(text)
        
        # 连接元件指令
        elif (
            any(keyword in text_lower for keyword in ["连接", "串联", "并联", "connect", "link", "wire"])
            # 或者：虽然没听出“连接”两个字，但句子里出现了两个元件ID，并且有“和/与/跟”这类连接词
            or (
                len(re.findall(r'([IRVW]\s*\d+|GND\s*\d*)', text, re.IGNORECASE)) >= 2
                and any(sep in text for sep in ["和", "与", "跟"])
            )
        ):
            return self._parse_connect_components(text)
        
        # 仿真指令
        elif any(keyword in text_lower for keyword in ["仿真", "运行", "开始", "run", "simulate", "start"]):
            return ParsedCommand(
                command_type=VoiceCommandType.RUN_SIMULATION,
                confidence=0.9,
                parameters={"type": "run_simulation"},
                original_text=text
            )

        # 修改元件参数指令（如：修改R1为两千欧、把V1改成5伏）
        elif any(keyword in text for keyword in ["修改", "改成", "改为", "设为", "设成"]):
            return self._parse_modify_component(text)
        
        # 验证定理指令
        elif any(keyword in text_lower for keyword in ["验证", "检查", "verify", "check", "kvl", "kcl"]):
            return self._parse_verify_theorem(text)

        # 仅“数值+欧/欧姆”（ASR 只识别到后半段时当作添加电阻，如 1000欧、一千欧）
        res_only = self._parse_resistance_only(text)
        if res_only:
            return res_only

        self.logger.warning(f"无法解析指令: {text}")
        return None

    def _normalize_component_id(self, raw_id: str) -> str:
        """ASR 常把 R 识别成 I：将 I1、I 1 规范为 R1"""
        if not raw_id:
            return raw_id
        s = raw_id.strip().upper().replace(" ", "")
        if s.startswith("I") and len(s) > 1 and s[1:].isdigit():
            return "R" + s[1:]
        return raw_id.strip().upper()

    def _parse_placement_row_col(self, s: str) -> Optional[int]:
        """行/列字符串 → 0-based 索引。支持「一、二、第一、第二」和「1、2、3」"""
        if not s:
            return None
        s = s.strip()
        chinese_nums = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
                       "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
        t = s.lstrip("第")
        if t in chinese_nums:
            val = chinese_nums[t]
        else:
            try:
                val = int(t)
            except ValueError:
                return None
        return max(0, val - 1)  # 1-based → 0-based

    def _parse_resistance_only(self, text: str) -> Optional[ParsedCommand]:
        """仅“数值+欧/欧姆”时解析为添加电阻（ASR 只识别到后半段时用）"""
        t = text.strip().rstrip("。.、,，")
        if not t or len(t) > 30:
            return None
        # 匹配：数字+可选千/k+欧 或 一千/两千+欧
        value = None
        if "千" in t or re.search(r"k\s*欧", t, re.IGNORECASE):
            m = re.search(r"([一二三四五六七八九十百两\d]+)\s*[千k]?\s*欧", t, re.IGNORECASE)
            if m:
                num_str = m.group(1).strip()
                chinese_nums = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
                if num_str in chinese_nums:
                    value = chinese_nums[num_str] * 1000
                else:
                    try:
                        value = float(num_str) * 1000
                    except ValueError:
                        value = 1000
        else:
            m = re.search(r"(\d+(?:\.\d+)?)\s*(?:欧姆?|ohm|Ω)?", t, re.IGNORECASE)
            if m and re.match(r"^\s*\d", t):
                value = float(m.group(1))
        if value is None:
            return None
        # 整句应主要为“数值+欧”，无其它明显动词
        if any(k in t for k in ["删除", "连接", "验证", "运行", "仿真", "添加", "加", "放置"]):
            return None
        n = self._component_counters.get("resistor", 1)
        comp_id = f"R{n}"
        self._component_counters["resistor"] = n + 1
        return ParsedCommand(
            command_type=VoiceCommandType.ADD_COMPONENT,
            confidence=0.75,
            parameters={
                "type": "add_component",
                "component_type": "resistor",
                "component_id": comp_id,
                "resistance": value,
            },
            original_text=text,
        )

    def _parse_add_component(self, text: str) -> ParsedCommand:
        """解析添加元件指令（支持中英文）"""
        parameters = {"type": "add_component"}
        text_lower = text.lower()
        
        # 提取元件类型（中英文）
        component_type = None
        if "电阻" in text or "resistor" in text_lower:
            component_type = "resistor"
        elif "电源" in text or "信号发生器" in text or "power" in text_lower or "source" in text_lower or "voltage" in text_lower:
            component_type = "power_source"
        elif "接地" in text or "ground" in text_lower or "gnd" in text_lower:
            component_type = "ground"
        
        parameters["component_type"] = component_type
        
        # 提取元件ID (如 R1, V1, GND1；ASR 可能把 R 识别成 I，统一规范)
        id_match = re.search(r'([IRVW]\s*\d+|GND\s*\d*)', text, re.IGNORECASE)
        if id_match:
            raw = id_match.group(1).replace(" ", "")
            parameters["component_id"] = self._normalize_component_id(raw)
        
        # 提取参数值 - 电阻（中英文）
        # 先处理中文数字（一千、两千等）
        if component_type == "resistor":
            # 匹配"一千欧姆"、"1千欧姆"、"1000欧姆"等
            if "千" in text or "k" in text_lower:
                # 提取千前面的数字
                resistance_match = re.search(r'([一二三四五六七八九十百两\d]+)\s*[千k]', text, re.IGNORECASE)
                if resistance_match:
                    num_str = resistance_match.group(1)
                    # 转换中文数字
                    chinese_nums = {'一': 1, '二': 2, '两': 2, '三': 3, '四': 4, '五': 5, 
                                   '六': 6, '七': 7, '八': 8, '九': 9, '十': 10}
                    if num_str in chinese_nums:
                        resistance = chinese_nums[num_str] * 1000
                    else:
                        try:
                            resistance = float(num_str) * 1000
                        except:
                            resistance = 1000  # 默认值
                    parameters["resistance"] = resistance
            else:
                # 匹配普通数字
                resistance_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:欧姆?|ohm|Ω)?', text, re.IGNORECASE)
                if resistance_match:
                    parameters["resistance"] = float(resistance_match.group(1))
        
        # 提取参数值 - 电压（中英文）
        voltage_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:伏特?|v(?:olt)?)', text, re.IGNORECASE)
        if voltage_match:
            parameters["voltage"] = float(voltage_match.group(1))

        # 方案 C：放置位置 - 第 N 行第 M 列（1-based 转 0-based）
        row_col_m = re.search(
            r'第?\s*([一二两三四五六七八九十\d]+)\s*行\s*第?\s*([一二两三四五六七八九十\d]+)\s*列',
            text
        )
        if row_col_m:
            r = self._parse_placement_row_col(row_col_m.group(1))
            c = self._parse_placement_row_col(row_col_m.group(2))
            if r is not None and c is not None:
                parameters["placement_row"] = r
                parameters["placement_col"] = c
        else:
            short_m = re.search(r'([一二两三四五六七八九十\d]+)\s*行\s*([一二两三四五六七八九十\d]+)\s*列', text)
            if short_m:
                r = self._parse_placement_row_col(short_m.group(1))
                c = self._parse_placement_row_col(short_m.group(2))
                if r is not None and c is not None:
                    parameters["placement_row"] = r
                    parameters["placement_col"] = c

        # 方案 C：放置位置 - 在某某元件 右/左/上/下 边（相对放置，优先于行/列）
        for direction_key, pattern in [
            ("right", r'在\s*([IRVW]\s*\d+|GND\s*\d*)\s*右\s*[侧边]'),
            ("left", r'在\s*([IRVW]\s*\d+|GND\s*\d*)\s*左\s*[侧边]'),
            ("front", r'在\s*([IRVW]\s*\d+|GND\s*\d*)\s*(?:上|前)\s*[侧边]'),
            ("back", r'在\s*([IRVW]\s*\d+|GND\s*\d*)\s*(?:下|后)\s*[侧边]'),
        ]:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                ref_id = self._normalize_component_id(m.group(1).replace(" ", ""))
                parameters["place_relative"] = {"ref_id": ref_id, "direction": direction_key}
                break

        # 检查是否需要澄清
        # 规则：
        # - 未识别出元件类型 => 一定需要澄清
        # - 接地(ground) 可以不带编号，后续会自动生成 GNDn => 不需要澄清
        # - 电阻(resistor) 没有阻值 => 需要澄清
        # - 电源(power_source) 没有电压 => 需要澄清
        if not component_type:
            needs_clarification = True
        elif component_type == "ground":
            needs_clarification = False
        elif component_type == "resistor" and "resistance" not in parameters:
            needs_clarification = True
        elif component_type == "power_source" and "voltage" not in parameters:
            needs_clarification = True
        else:
            needs_clarification = False
        
        return ParsedCommand(
            command_type=VoiceCommandType.ADD_COMPONENT,
            confidence=0.8 if not needs_clarification else 0.5,
            parameters=parameters,
            original_text=text,
            needs_clarification=needs_clarification,
            clarification_question="请指定元件参数 / Please specify component parameters" if needs_clarification else None
        )

    def _parse_modify_component(self, text: str) -> ParsedCommand:
        """解析修改元件参数指令：修改R1为两千欧、把V1改成5伏 等"""
        parameters = {"type": "modify_component"}
        raw_ids = re.findall(r'([IRVW]\s*\d+|GND\s*\d*)', text, re.IGNORECASE)
        if not raw_ids:
            return ParsedCommand(
                command_type=VoiceCommandType.MODIFY_COMPONENT,
                confidence=0.3,
                parameters=parameters,
                original_text=text,
                needs_clarification=True,
                clarification_question="请指定要修改的元件，如 R1、V1"
            )
        comp_id = self._normalize_component_id(raw_ids[0].replace(" ", ""))
        parameters["component_id"] = comp_id

        # 阻值：两千欧、2000欧、2k欧、1.5千欧
        resistance = None
        if "欧" in text or "ohm" in text.lower() or "Ω" in text:
            if "千" in text or "k" in text.lower():
                m = re.search(r'([一二三四五六七八九十百两\d.]+)\s*[千k]?\s*欧', text, re.IGNORECASE)
                if m:
                    num_str = m.group(1).strip()
                    chinese_nums = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
                    if num_str in ("两千", "一千"):
                        resistance = 2000 if num_str == "两千" else 1000
                    elif num_str in chinese_nums:
                        resistance = chinese_nums[num_str] * 1000
                    else:
                        try:
                            resistance = float(num_str) * 1000
                        except ValueError:
                            resistance = 1000
            else:
                m = re.search(r'(\d+(?:\.\d+)?)\s*(?:欧姆?|ohm|Ω)?', text, re.IGNORECASE)
                if m:
                    resistance = float(m.group(1))
        if resistance is not None:
            parameters["resistance"] = resistance
            return ParsedCommand(
                command_type=VoiceCommandType.MODIFY_COMPONENT,
                confidence=0.85,
                parameters=parameters,
                original_text=text
            )

        # 电压：5伏、5V、10伏特
        voltage = None
        if "伏" in text or "v" in text.lower() or "volt" in text.lower():
            m = re.search(r'(\d+(?:\.\d+)?)\s*(?:伏特?|v(?:olt)?)', text, re.IGNORECASE)
            if m:
                voltage = float(m.group(1))
        if voltage is not None:
            parameters["voltage"] = voltage
            return ParsedCommand(
                command_type=VoiceCommandType.MODIFY_COMPONENT,
                confidence=0.85,
                parameters=parameters,
                original_text=text
            )

        return ParsedCommand(
            command_type=VoiceCommandType.MODIFY_COMPONENT,
            confidence=0.4,
            parameters=parameters,
            original_text=text,
            needs_clarification=True,
            clarification_question="请指定要修改的参数值，如 两千欧、5伏"
        )
    
    def _parse_connect_components(self, text: str) -> ParsedCommand:
        """解析连接元件指令（支持中英文）"""
        parameters = {"type": "connect_components"}

        # 提取元件ID（支持 I1/I 1 规范为 R1）
        raw_ids = re.findall(r'([IRVW]\s*\d+|GND\s*\d*)', text, re.IGNORECASE)
        component_ids = [self._normalize_component_id(x.replace(" ", "")) for x in raw_ids]
        
        if len(component_ids) >= 2:
            parameters["component1_id"] = component_ids[0]
            parameters["component2_id"] = component_ids[1]
        
        # 判断连接类型（中英文）
        text_lower = text.lower()
        connection_type = "series"  # 默认串联
        if "并联" in text or "parallel" in text_lower:
            connection_type = "parallel"
        parameters["connection_type"] = connection_type
        
        # 检查是否有足够的元件ID
        needs_clarification = len(component_ids) < 2
        
        return ParsedCommand(
            command_type=VoiceCommandType.CONNECT_COMPONENTS,
            confidence=0.85 if not needs_clarification else 0.4,
            parameters=parameters,
            original_text=text,
            needs_clarification=needs_clarification,
            clarification_question="请指定要连接的两个元件ID" if needs_clarification else None
        )
    
    def _parse_verify_theorem(self, text: str) -> ParsedCommand:
        """解析验证定理指令"""
        parameters = {"type": "verify_theorem"}
        
        # 提取定理类型
        theorem = "kvl"  # 默认KVL
        if "kcl" in text.lower() or "电流定律" in text:
            theorem = "kcl"
        elif "kvl" in text.lower() or "电压定律" in text:
            theorem = "kvl"
        
        parameters["theorem"] = theorem
        
        return ParsedCommand(
            command_type=VoiceCommandType.VERIFY_THEOREM,
            confidence=0.9,
            parameters=parameters,
            original_text=text
        )
    
    def _parse_delete_component(self, text: str) -> ParsedCommand:
        """解析删除元件指令"""
        parameters = {"type": "delete_component"}

        # 提取元件ID（支持 I1/I 1 规范为 R1）
        id_match = re.search(r'([IRVW]\s*\d+|GND\s*\d*|[A-Za-z]+\s*\d+)', text)
        if id_match:
            raw = id_match.group(1).replace(" ", "")
            parameters["component_id"] = self._normalize_component_id(raw)
        
        # 判断是删除元件还是删除导线
        if "导线" in text or "连线" in text or text.startswith("W"):
            parameters["delete_type"] = "wire"
        else:
            parameters["delete_type"] = "component"
        
        needs_clarification = "component_id" not in parameters
        
        return ParsedCommand(
            command_type=VoiceCommandType.DELETE_COMPONENT,
            confidence=0.85 if not needs_clarification else 0.4,
            parameters=parameters,
            original_text=text,
            needs_clarification=needs_clarification,
            clarification_question="请指定要删除的元件ID" if needs_clarification else None
        )
    
    def _validate_command(self, command: ParsedCommand) -> bool:
        """校验指令"""
        try:
            # 检查元件ID唯一性
            if command.command_type == VoiceCommandType.ADD_COMPONENT:
                component_id = command.parameters.get("component_id")
                if component_id and self.data_layer.get_component(component_id):
                    self.logger.warning(f"元件ID {component_id} 已存在")
                    return False
            
            # 检查参数合理性
            if "resistance" in command.parameters:
                if command.parameters["resistance"] <= 0:
                    self.logger.warning("电阻值必须大于0")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"指令校验错误: {e}")
            return False
    
    def _store_command_context(self, original_text: str, parsed_command: ParsedCommand) -> None:
        """存储指令上下文"""
        context = VoiceCommandContext(
            command_id=f"cmd_{len(self.data_layer.voice_context) + 1}",
            timestamp=str(datetime.now()),
            original_text=original_text,
            parsed_command=parsed_command.parameters,
            execution_result="success" if not parsed_command.needs_clarification else "needs_clarification",
            referenced_components=self._extract_referenced_components(parsed_command)
        )
        
        self.data_layer.add_voice_context(context)
    
    def _extract_referenced_components(self, command: ParsedCommand) -> List[str]:
        """提取指令中引用的元件ID"""
        referenced = []
        
        if command.command_type == VoiceCommandType.ADD_COMPONENT:
            comp_id = command.parameters.get("component_id")
            if comp_id:
                referenced.append(comp_id)
        
        elif command.command_type == VoiceCommandType.CONNECT_COMPONENTS:
            comp1_id = command.parameters.get("component1_id")
            comp2_id = command.parameters.get("component2_id")
            if comp1_id:
                referenced.append(comp1_id)
            if comp2_id:
                referenced.append(comp2_id)
        
        return referenced
    
    def generate_feedback(self, result_data: Dict[str, Any]) -> str:
        """生成自然语言反馈"""
        try:
            if result_data.get("type") == "kvl_verification":
                if result_data.get("passed"):
                    return f"基尔霍夫电压定律验证成功，回路总电压为 {result_data.get('total_voltage', 0):.4f}V，符合定律要求。"
                else:
                    return f"基尔霍夫电压定律验证失败，回路总电压为 {result_data.get('total_voltage', 0):.4f}V。"
            
            elif result_data.get("type") == "kcl_verification":
                if result_data.get("passed"):
                    return f"基尔霍夫电流定律验证成功，节点电流守恒。"
                else:
                    return f"基尔霍夫电流定律验证失败。"
            
            return "操作完成"
            
        except Exception as e:
            self.logger.error(f"反馈生成错误: {e}")
            return "操作已完成"
    
    def load_ernie_model(self, model_path: str = None) -> bool:
        """加载或重新加载ERNIE模型"""
        try:
            if not _ERNIE_AVAILABLE:
                self.logger.error("ERNIE模块不可用")
                return False
            
            # 如果指定了路径，创建新实例
            if model_path:
                from ..ernie_integration_fixed import ElectricLabERNIE
                self.ernie_model = ElectricLabERNIE(model_path)
            else:
                self.ernie_model = get_ernie_instance()
            
            # 加载模型
            if self.ernie_model.load_model():
                self.logger.info("ERNIE模型加载成功")
                return True
            else:
                self.logger.error("ERNIE模型加载失败")
                self.ernie_model = None
                return False
            
        except Exception as e:
            self.logger.error(f"ERNIE模型加载失败: {e}")
            self.ernie_model = None
            return False
    
    # ========== 指令执行方法（与VR层联动） ==========
    
    def parse_command_only(self, text: str) -> Optional[ParsedCommand]:
        """
        仅解析语音指令（不执行），可在后台线程调用，避免阻塞VR主线程。
        返回 ParsedCommand 或 None。
        """
        raw_text = (text or "").strip()
        if not raw_text:
            return None
        text = self._normalize_asr_text(raw_text)
        return self._parse_voice_command(text)
    
    def execute_command(self, text: str, parsed_command: Optional[ParsedCommand] = None) -> Dict[str, Any]:
        """
        解析并执行语音指令，返回执行结果。
        若传入 parsed_command 则跳过解析（解析已在后台完成），仅执行，避免阻塞菜单/迷你界面。
        """
        result = {
            "success": False,
            "command_type": None,
            "message": "",
            "data": {}
        }
        
        try:
            raw_text = (text or "").strip()
            if not raw_text and not parsed_command:
                result["message"] = "未识别到有效文字"
                return result
            if not raw_text and parsed_command:
                raw_text = parsed_command.original_text
            result["recognized_text"] = raw_text  # 供界面显示
            
            # 1. 解析指令（若未传入则在此解析）
            if parsed_command is None:
                text = self._normalize_asr_text(raw_text)
                parsed_command = self._parse_voice_command(text)
            if not parsed_command:
                result["message"] = "未识别为电学指令，请说：添加电阻、运行仿真、连接R1和V1 等"
                return result
            
            result["command_type"] = parsed_command.command_type.value
            
            # 2. 检查是否需要澄清
            if parsed_command.needs_clarification:
                result["message"] = parsed_command.clarification_question
                result["needs_clarification"] = True
                return result
            
            # 3. 校验指令
            if not self._validate_command(parsed_command):
                result["message"] = "指令校验失败"
                return result
            
            # 4. 执行指令（在主线程，操作 Ursina 实体）
            exec_result = self._dispatch_command(parsed_command)
            result.update(exec_result)
            
            # 5. 存储上下文
            if result["success"]:
                self._store_command_context(parsed_command.original_text, parsed_command)

            return result
            
        except Exception as e:
            self.logger.error(f"指令执行错误: {e}")
            result["message"] = f"执行错误: {str(e)}"
            if "recognized_text" not in result:
                result["recognized_text"] = raw_text if raw_text else ""
            return result
    
    def _dispatch_command(self, command: ParsedCommand) -> Dict[str, Any]:
        """分发并执行指令"""
        handlers = {
            VoiceCommandType.ADD_COMPONENT: self._execute_add_component,
            VoiceCommandType.CONNECT_COMPONENTS: self._execute_connect_components,
            VoiceCommandType.DELETE_COMPONENT: self._execute_delete_component,
            VoiceCommandType.RUN_SIMULATION: self._execute_run_simulation,
            VoiceCommandType.VERIFY_THEOREM: self._execute_verify_theorem,
            VoiceCommandType.MODIFY_COMPONENT: self._execute_modify_component,
        }
        
        handler = handlers.get(command.command_type)
        if handler:
            return handler(command.parameters)
        
        return {"success": False, "message": f"未知指令类型 / Unknown command: {command.command_type.value}"}
    
    def _execute_modify_component(self, params: Dict) -> Dict[str, Any]:
        """执行修改元件参数指令（通过 VR 层更新实体与数据）"""
        comp_id = params.get("component_id")
        if not comp_id:
            return {"success": False, "message": "未指定元件"}
        if self.vr_layer:
            success = self.vr_layer.execute_voice_command("modify_component", params)
            if success:
                return {"success": True, "message": f"已修改 {comp_id}", "data": {"component_id": comp_id}}
            return {"success": False, "message": f"修改失败: {comp_id}"}
        return {"success": False, "message": "VR 层未就绪"}
    
    def _execute_add_component(self, params: Dict) -> Dict[str, Any]:
        """执行添加元件指令"""
        comp_type = params.get("component_type")
        comp_id = params.get("component_id")
        
        # 自动生成ID（如果未指定）
        if not comp_id:
            prefix_map = {'resistor': 'R', 'power_source': 'V', 'ground': 'GND'}
            prefix = prefix_map.get(comp_type, 'COMP')
            comp_id = f"{prefix}{self._component_counters.get(comp_type, 1)}"
            self._component_counters[comp_type] = self._component_counters.get(comp_type, 1) + 1
            params["component_id"] = comp_id
        
        # 设置默认位置（VR层会自动调整）
        if "position" not in params:
            params["position"] = (0, 0.8, 0)
        
        # 通过VR层添加元件（如果已绑定）
        if self.vr_layer:
            success = self.vr_layer.execute_voice_command("add_component", params)
            if success:
                return {
                    "success": True,
                    "message": f"已添加元件: {comp_id}",
                    "data": {"component_id": comp_id}
                }
            else:
                return {"success": False, "message": f"VR层添加元件失败: {comp_id}"}
        else:
            # 无VR层时，直接添加到数据层
            self.logger.warning("VR层未绑定，仅添加到数据层")
            
            # 创建数据层元件对象
            from ..data.data_layer import Resistor, PowerSource, Ground
            
            position = params.get("position", (0, 0.8, 0))
            
            if comp_type == "resistor":
                resistance = params.get("resistance", 1000)
                component = Resistor(id=comp_id, position=position, resistance=resistance)
            elif comp_type == "power_source":
                voltage = params.get("voltage", 5)
                component = PowerSource(id=comp_id, position=position, voltage=voltage)
            elif comp_type == "ground":
                component = Ground(id=comp_id, position=position)
            else:
                return {"success": False, "message": f"未知元件类型: {comp_type}"}
            
            # 添加到数据层
            if self.data_layer.add_component(component):
                return {
                    "success": True,
                    "message": f"已添加元件到数据层: {comp_id}（VR层未绑定）",
                    "data": {"component_id": comp_id}
                }
            else:
                return {"success": False, "message": f"添加元件到数据层失败: {comp_id}"}
    
    def _execute_connect_components(self, params: Dict) -> Dict[str, Any]:
        """执行连接元件指令"""
        comp1_id = params.get("component1_id")
        comp2_id = params.get("component2_id")
        
        if not comp1_id or not comp2_id:
            return {"success": False, "message": "请指定要连接的两个元件ID / Please specify two component IDs"}
        
        # 检查元件是否存在
        comp1 = self.data_layer.get_component(comp1_id)
        comp2 = self.data_layer.get_component(comp2_id)
        
        if not comp1:
            return {"success": False, "message": f"元件 {comp1_id} 不存在 / Component {comp1_id} not found"}
        if not comp2:
            return {"success": False, "message": f"元件 {comp2_id} 不存在 / Component {comp2_id} not found"}
        
        # 生成导线ID
        wire_id = f"W{self._wire_counter}"
        self._wire_counter += 1
        params["wire_id"] = wire_id
        
        # 通过VR层连接元件
        if self.vr_layer:
            success = self.vr_layer.execute_voice_command("connect_components", params)
            if success:
                return {
                    "success": True,
                    "message": f"已连接: {comp1_id} <-> {comp2_id}",
                    "data": {"wire_id": wire_id}
                }
            else:
                return {"success": False, "message": f"VR层连接失败: {comp1_id} <-> {comp2_id}"}
        else:
            # 无VR层时，直接添加到数据层
            from ..data.data_layer import Connection, ComponentType
            from ..data.data_layer import (
                TERMINAL_POSITIVE, TERMINAL_NEGATIVE, TERMINAL_1, TERMINAL_2, TERMINAL_COMMON
            )
            # 端子化仿真要求每条连接带端子信息：无VR层时用默认规则补全
            def terminals_for(c):
                if c.type == ComponentType.POWER_SOURCE:
                    return [TERMINAL_POSITIVE, TERMINAL_NEGATIVE]
                if c.type == ComponentType.RESISTOR:
                    return [TERMINAL_1, TERMINAL_2]
                if c.type == ComponentType.GROUND:
                    return [TERMINAL_COMMON]
                return []
            def used_terminals(cid: str):
                used = set()
                for cc in self.data_layer.connections.values():
                    if getattr(cc, 'terminal1_id', None) and getattr(cc, 'terminal2_id', None):
                        if cc.component1_id == cid:
                            used.add(cc.terminal1_id)
                        if cc.component2_id == cid:
                            used.add(cc.terminal2_id)
                return used
            def choose_terminal(cid: str):
                c = self.data_layer.get_component(cid)
                if not c:
                    return None
                ts = terminals_for(c)
                if not ts:
                    return None
                used = used_terminals(cid)
                for t in ts:
                    if t not in used:
                        return t
                return ts[0]
            
            connection = Connection(
                id=wire_id,
                component1_id=comp1_id,
                component2_id=comp2_id,
                connection_point1=comp1.position,
                connection_point2=comp2.position
            )
            connection.terminal1_id = choose_terminal(comp1_id)
            connection.terminal2_id = choose_terminal(comp2_id)
            
            if self.data_layer.add_connection(connection):
                return {
                    "success": True,
                    "message": f"连接已添加到数据层（VR层未绑定）: {comp1_id} <-> {comp2_id}",
                    "data": {"wire_id": wire_id}
                }
            else:
                return {"success": False, "message": f"添加连接到数据层失败"}
    
    def _execute_delete_component(self, params: Dict) -> Dict[str, Any]:
        """执行删除元件/导线指令"""
        comp_id = params.get("component_id")
        delete_type = params.get("delete_type", "component")
        
        if not comp_id:
            return {"success": False, "message": "请指定要删除的元件或导线ID"}
        
        # 通过VR层删除
        if self.vr_layer:
            if delete_type == "wire":
                success = self.vr_layer.execute_voice_command("delete_wire", {"wire_id": comp_id})
            else:
                success = self.vr_layer.execute_voice_command("delete_component", {"component_id": comp_id})
            
            if success:
                return {
                    "success": True,
                    "message": f"已删除: {comp_id}",
                    "data": {"deleted_id": comp_id}
                }
            else:
                return {"success": False, "message": f"删除失败: {comp_id}"}
        else:
            # 无VR层时，仅从数据层删除
            if delete_type == "wire":
                self.data_layer.remove_connection(comp_id)
            else:
                self.data_layer.remove_component(comp_id)
            return {
                "success": True,
                "message": f"已从数据层删除: {comp_id}（VR层未绑定）",
                "data": {"deleted_id": comp_id}
            }
    
    def _execute_run_simulation(self, params: Dict) -> Dict[str, Any]:
        """执行仿真指令"""
        if self.vr_layer and hasattr(self.vr_layer, 'circuit_sim_layer') and self.vr_layer.circuit_sim_layer:
            # 通过电路仿真层运行仿真
            try:
                result = self.vr_layer.circuit_sim_layer.run_simulation()
                return {
                    "success": True,
                    "message": "仿真运行完成",
                    "data": result
                }
            except Exception as e:
                return {"success": False, "message": f"仿真运行失败: {e}"}
        else:
            return {
                "success": False,
                "message": "仿真层未初始化"
            }
    
    def _execute_verify_theorem(self, params: Dict) -> Dict[str, Any]:
        """执行定理验证指令"""
        theorem = params.get("theorem", "kvl")
        
        if self.vr_layer and hasattr(self.vr_layer, 'circuit_sim_layer') and self.vr_layer.circuit_sim_layer:
            try:
                if theorem == "kvl":
                    result = self.vr_layer.circuit_sim_layer.verify_kvl()
                else:
                    result = self.vr_layer.circuit_sim_layer.verify_kcl()
                
                feedback = self.generate_feedback({"type": f"{theorem}_verification", **result})
                return {
                    "success": True,
                    "message": feedback,
                    "data": result
                }
            except Exception as e:
                return {"success": False, "message": f"定理验证失败: {e}"}
        else:
            return {
                "success": False,
                "message": "仿真层未初始化，无法验证定理"
            }
    
    def test_voice_system(self) -> None:
        """测试语音系统"""
        test_phrases = [
            "添加一个1千欧的电阻R1",
            "添加5伏电源V1",
            "连接R1和V1",
            "添加接地GND1",
            "删除电阻R1",
            "运行仿真",
            "验证基尔霍夫电压定律"
        ]
        
        for phrase in test_phrases:
            print(f"测试指令: {phrase}")
            parsed = self._parse_with_rules(phrase)
            if parsed:
                print(f"  类型: {parsed.command_type.value}")
                print(f"  参数: {parsed.parameters}")
                print(f"  置信度: {parsed.confidence}")
            print("-" * 50)
