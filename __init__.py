"""
AI语音交互层模块
负责语音识别、指令解析和自然语言交互
"""

# 导入核心类和枚举
from .ai_voice_layer import (
    # 枚举
    VoiceCommandType,
    
    # 数据类（使用 @dataclass）
    ParsedCommand,
    
    # 主类
    AIVoiceLayer
)

# 公开的API接口
__all__ = [
    # 枚举
    'VoiceCommandType',
    
    # 数据类
    'ParsedCommand',
    
    # 主类
    'AIVoiceLayer'
]