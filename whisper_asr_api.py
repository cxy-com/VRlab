# -*- coding: utf-8 -*-
"""
Whisper 本地语音识别 API 封装
用于将语音转换为文本（离线模式）
"""

import io
import logging
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)

# 全局模型实例（延迟加载）
_whisper_model = None
_model_loaded = False


class WhisperASRAPI:
    """Whisper 语音识别 API 封装"""
    
    def __init__(self, model_size: str = "small", device: str = "cpu"):
        """
        初始化 Whisper API
        
        Args:
            model_size: 模型大小 "tiny", "base", "small", "medium", "large"
            device: 运行设备 "cpu" 或 "cuda"
        """
        self.model_size = model_size
        self.device = device
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """加载 Whisper 模型"""
        global _whisper_model, _model_loaded
        
        if _model_loaded and _whisper_model is not None:
            self.model = _whisper_model
            logger.info(f"复用已加载的 Whisper 模型: {self.model_size}")
            return
        
        try:
            import whisper
            logger.info(f"正在加载 Whisper 模型: {self.model_size}...")
            self.model = whisper.load_model(self.model_size, device=self.device)
            _whisper_model = self.model
            _model_loaded = True
            logger.info(f"Whisper 模型加载成功: {self.model_size}")
        except Exception as e:
            logger.error(f"Whisper 模型加载失败: {e}")
            self.model = None
    
    def recognize_audio(self, audio_data: bytes, sample_rate: int = 16000) -> Optional[str]:
        """
        识别音频
        
        Args:
            audio_data: 音频数据（PCM 格式）
            sample_rate: 采样率，默认 16000
            
        Returns:
            识别的文本，失败返回 None
        """
        if self.model is None:
            logger.error("Whisper 模型未加载")
            return None
        
        try:
            # 将 bytes 转换为 numpy 数组
            audio_np = np.frombuffer(audio_data, dtype=np.int16)
            audio_np = audio_np.astype(np.float32) / 32768.0  # 归一化到 [-1, 1]
            
            # 如果采样率不是 16k，需要重采样
            if sample_rate != 16000:
                try:
                    import librosa
                    audio_np = librosa.resample(audio_np, orig_sr=sample_rate, target_sr=16000)
                except ImportError:
                    logger.warning("librosa 未安装，使用原始采样率")
            
            # 执行识别
            result = self.model.transcribe(audio_np, language="zh", fp16=False)
            text = result.get("text", "").strip()
            
            logger.info(f"Whisper 识别完成: {text}")
            return text
            
        except Exception as e:
            logger.error(f"Whisper 识别失败: {e}")
            return None


# 全局 API 实例
_api_instance = None


def get_whisper_api(model_size: str = "small", device: str = "cpu") -> WhisperASRAPI:
    """获取 Whisper API 单例"""
    global _api_instance
    if _api_instance is None:
        _api_instance = WhisperASRAPI(model_size, device)
    return _api_instance


def recognize_speech(audio_data: bytes, sample_rate: int = 16000, 
                     model_size: str = "small") -> Optional[str]:
    """
    识别语音（简化接口）
    
    Args:
        audio_data: 音频数据（PCM 格式）
        sample_rate: 采样率，默认 16000
        model_size: 模型大小，默认 "small"
    
    Returns:
        识别的文本
    """
    api = get_whisper_api(model_size)
    return api.recognize_audio(audio_data, sample_rate)


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    print("Whisper 语音识别 API 模块")
    print("使用方法：")
    print("  from whisper_asr_api import recognize_speech")
    print("  text = recognize_speech(audio_data)")
