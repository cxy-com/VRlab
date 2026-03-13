"""
微调 Whisper 语音识别 API - 专用于电路指令
基于在 AutoDL 上微调的 whisper-small 模型
"""

import os
import logging
import numpy as np
import torch
from typing import Optional

logger = logging.getLogger(__name__)

# 全局模型实例（延迟加载，避免重复加载）
_processor = None
_model = None
_device = None
_model_loaded = False

# 模型路径（优先级：本地项目目录 > 当前目录）
MODEL_PATHS = [
    r"D:\python\whisper_finetuned_circuit\final",
    r".\whisper_finetuned_circuit\final",
    "./whisper_finetuned_circuit/final",
]


class FinetunedWhisperAPI:
    """微调 Whisper 语音识别 API - 专用于电路指令识别"""
    
    def __init__(self, model_path: Optional[str] = None, device: str = "auto"):
        """
        初始化微调 Whisper API
        
        Args:
            model_path: 微调模型路径，None 则自动查找
            device: "cpu", "cuda" 或 "auto"
        """
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # 查找模型路径
        self.model_path = model_path or self._find_model_path()
        self.model = None
        self.processor = None
        
        self._load_model()
    
    def _find_model_path(self) -> Optional[str]:
        """自动查找模型路径"""
        for path in MODEL_PATHS:
            if os.path.exists(path) and os.path.exists(os.path.join(path, "model.safetensors")):
                logger.info(f"找到微调模型: {path}")
                return path
        logger.warning("未找到微调模型，请检查路径")
        return None
    
    def _load_model(self):
        """加载微调模型"""
        global _processor, _model, _device, _model_loaded
        
        # 如果已经加载过，复用全局实例
        if _model_loaded and _model is not None:
            self.model = _model
            self.processor = _processor
            self.device = _device
            logger.info("复用已加载的微调 Whisper 模型")
            return
        
        if self.model_path is None:
            logger.error("未找到微调模型路径")
            return
        
        try:
            from transformers import WhisperProcessor, WhisperForConditionalGeneration
            
            logger.info(f"正在加载微调模型: {self.model_path}")
            logger.info(f"使用设备: {self.device}")
            
            self.processor = WhisperProcessor.from_pretrained(self.model_path)
            self.model = WhisperForConditionalGeneration.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            
            # 保存到全局
            _processor = self.processor
            _model = self.model
            _device = self.device
            _model_loaded = True
            
            logger.info("✅ 微调 Whisper 模型加载成功")
            
        except Exception as e:
            logger.error(f"微调模型加载失败: {e}")
            self.model = None
            self.processor = None
    
    def is_available(self) -> bool:
        """检查模型是否可用"""
        return self.model is not None and self.processor is not None
    
    def recognize_pcm(self, pcm_data: bytes, sample_rate: int = 16000) -> Optional[str]:
        """
        识别 PCM 音频数据（与现有接口兼容）
        
        Args:
            pcm_data: PCM 音频数据（int16）
            sample_rate: 采样率
            
        Returns:
            识别的文本
        """
        if not self.is_available():
            return None
        
        try:
            # 将 bytes 转换为 numpy 数组
            audio_np = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # 重采样（如果需要）
            if sample_rate != 16000:
                try:
                    import librosa
                    audio_np = librosa.resample(audio_np, orig_sr=sample_rate, target_sr=16000)
                    logger.debug(f"音频从 {sample_rate}Hz 重采样到 16000Hz")
                except ImportError:
                    logger.warning("librosa 未安装，使用原始采样率")
            
            # 使用 transformers 进行识别
            inputs = self.processor(
                audio_np,
                sampling_rate=16000,
                return_tensors="pt"
            )
            input_features = inputs.input_features.to(self.device)
            
            with torch.no_grad():
                predicted_ids = self.model.generate(input_features)
            
            text = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            text = text.strip()
            
            logger.info(f"微调模型识别: {text}")
            return text
            
        except Exception as e:
            logger.error(f"微调模型识别失败: {e}")
            return None
    
    def recognize_file(self, audio_path: str) -> Optional[str]:
        """从文件识别"""
        try:
            import soundfile as sf
            audio, sr = sf.read(audio_path)
            
            # 转换为 PCM bytes
            audio_int16 = (audio * 32768).astype(np.int16)
            pcm_data = audio_int16.tobytes()
            
            return self.recognize_pcm(pcm_data, sr)
        except Exception as e:
            logger.error(f"文件识别失败: {e}")
            return None


# 全局 API 实例（单例）
_api_instance = None

def get_finetuned_api(model_path: Optional[str] = None, device: str = "auto") -> FinetunedWhisperAPI:
    """获取微调 API 单例"""
    global _api_instance
    if _api_instance is None:
        _api_instance = FinetunedWhisperAPI(model_path, device)
    return _api_instance

def recognize_speech(pcm_data: bytes, sample_rate: int = 16000, 
                     model_path: Optional[str] = None) -> Optional[str]:
    """
    识别语音（简化接口，与现有代码兼容）
    """
    api = get_finetuned_api(model_path)
    if not api.is_available():
        return None
    return api.recognize_pcm(pcm_data, sample_rate)


if __name__ == "__main__":
    # 测试
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    print("=" * 50)
    print("微调 Whisper 模型测试")
    print("=" * 50)
    
    api = FinetunedWhisperAPI()
    if api.is_available():
        print("✅ 微调模型加载成功！")
        print(f"模型路径: {api.model_path}")
        print(f"使用设备: {api.device}")
        print(f"处理器: {api.processor.__class__.__name__}")
        print(f"模型: {api.model.__class__.__name__}")
    else:
        print("❌ 模型加载失败")
        print("请确保模型文件在 D:\\python\\whisper_finetuned_circuit\\final\\")
