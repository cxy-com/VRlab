# -*- coding: utf-8 -*-
"""
讯飞语音识别（语音听写）API封装
用于将语音转换为文本
"""

import json
import ssl
import hmac
import base64
import hashlib
from datetime import datetime
from time import mktime
from urllib.parse import urlencode, urlparse
from wsgiref.handlers import format_date_time
import websocket
import logging
from typing import Optional
import os

logger = logging.getLogger(__name__)


class XFYunASRAPI:
    """讯飞语音识别API封装"""
    
    def __init__(self, config_path: str = None):
        """
        初始化讯飞语音识别API
        
        Args:
            config_path: 配置文件路径，默认使用xfyun_spark_config.json
        """
        if config_path is None:
            # 使用与星火大模型相同的配置文件
            config_path = os.path.join(os.path.dirname(__file__), 'xfyun_spark_config.json')
        
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        self.appid = self.config['appid']
        self.api_secret = self.config['api_secret']
        self.api_key = self.config['api_key']
        
        # 语音识别API地址
        self.api_url = "wss://iat-api.xfyun.cn/v2/iat"
        
        # 响应结果
        self.response_text = ""
        self.is_complete = False
        
        logger.info(f"讯飞语音识别API初始化完成 - APPID: {self.appid}")
    
    def create_url(self) -> str:
        """生成鉴权URL"""
        url_parts = urlparse(self.api_url)
        host = url_parts.netloc
        path = url_parts.path
        
        # 生成RFC1123格式的时间戳
        now = datetime.now()
        date = format_date_time(mktime(now.timetuple()))
        
        # 拼接签名原文
        signature_origin = f"host: {host}\ndate: {date}\nGET {path} HTTP/1.1"
        
        # 进行hmac-sha256加密
        signature_sha = hmac.new(
            self.api_secret.encode('utf-8'),
            signature_origin.encode('utf-8'),
            digestmod=hashlib.sha256
        ).digest()
        
        # Base64编码
        signature_sha_base64 = base64.b64encode(signature_sha).decode(encoding='utf-8')
        
        # 构建authorization
        authorization_origin = (
            f'api_key="{self.api_key}", '
            f'algorithm="hmac-sha256", '
            f'headers="host date request-line", '
            f'signature="{signature_sha_base64}"'
        )
        authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(encoding='utf-8')
        
        # 生成鉴权URL
        params = {
            "authorization": authorization,
            "date": date,
            "host": host
        }
        
        return f"{self.api_url}?{urlencode(params)}"
    
    def recognize_audio(self, audio_data: bytes, audio_format: str = "audio/L16;rate=16000") -> Optional[str]:
        """
        识别音频
        
        Args:
            audio_data: 音频数据（PCM格式）
            audio_format: 音频格式，默认16k采样率PCM
        
        Returns:
            识别的文本，失败返回None
        """
        self.response_text = ""
        self.is_complete = False
        
        try:
            # 创建WebSocket连接
            ws_url = self.create_url()
            ws = websocket.WebSocketApp(
                ws_url,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
                on_open=lambda ws: self._on_open(ws, audio_data, audio_format)
            )
            
            # 运行WebSocket（阻塞）
            ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
            
            return self.response_text if self.response_text else None
            
        except Exception as e:
            logger.error(f"讯飞语音识别API调用失败: {e}")
            return None
    
    def _on_open(self, ws, audio_data: bytes, audio_format: str):
        """WebSocket连接建立时发送音频数据"""
        # 构建请求参数
        data = {
            "common": {
                "app_id": self.appid
            },
            "business": {
                "language": "zh_cn",  # 中文
                "domain": "iat",  # 语音听写
                "accent": "mandarin",  # 普通话
                "vad_eos": 2000,  # 静音检测时长（ms）
                "dwa": "wpgs"  # 动态修正
            },
            "data": {
                "status": 2,  # 2表示最后一帧
                "format": audio_format,
                "encoding": "raw",
                "audio": base64.b64encode(audio_data).decode('utf-8')
            }
        }

        ws.send(json.dumps(data))
        logger.debug(f"发送音频数据: {len(audio_data)} bytes")
    
    def _on_message(self, ws, message):
        """接收响应消息"""
        try:
            data = json.loads(message)
            code = data['code']
            
            if code != 0:
                logger.error(f"API返回错误: {data['message']}")
                ws.close()
                return
            
            # 提取识别结果
            if 'data' in data:
                result = data['data'].get('result', {})
                ws_list = result.get('ws', [])
                
                # 拼接识别文本
                for ws_item in ws_list:
                    for cw in ws_item.get('cw', []):
                        word = cw.get('w', '')
                        self.response_text += word
            
            # 判断是否结束
            if data.get('data', {}).get('status') == 2:
                self.is_complete = True
                logger.info(f"识别完成: {self.response_text}")
                ws.close()
                
        except Exception as e:
            logger.error(f"解析响应失败: {e}")
            ws.close()
    
    def _on_error(self, ws, error):
        """错误处理"""
        logger.error(f"WebSocket错误: {error}")
    
    def _on_close(self, ws, close_status_code, close_msg):
        """连接关闭"""
        logger.debug("WebSocket连接已关闭")


# 简化的调用接口
def recognize_speech(audio_data: bytes, audio_format: str = "audio/L16;rate=16000", 
                     config_path: str = None) -> Optional[str]:
    """
    识别语音（简化接口）
    
    Args:
        audio_data: 音频数据（PCM格式）
        audio_format: 音频格式
        config_path: 配置文件路径
    
    Returns:
        识别的文本
    """
    api = XFYunASRAPI(config_path)
    return api.recognize_audio(audio_data, audio_format)


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    # 注意：需要实际的音频数据进行测试
    print("讯飞语音识别API模块")
    print("使用方法：")
    print("  from xfyun_asr_api import recognize_speech")
    print("  text = recognize_speech(audio_data)")
