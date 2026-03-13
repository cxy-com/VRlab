# -*- coding: utf-8 -*-
"""
讯飞星火大模型API调用模块
用于将用户语音指令转换为结构化JSON
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
from typing import Optional, Dict, Any
import os

logger = logging.getLogger(__name__)


class XFYunSparkAPI:
    """讯飞星火大模型API封装"""
    
    def __init__(self, config_path: str = None):
        """
        初始化讯飞星火API
        
        Args:
            config_path: 配置文件路径，默认为当前目录下的xfyun_spark_config.json
        """
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), 'xfyun_spark_config.json')
        
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        self.appid = self.config['appid']
        self.api_secret = self.config['api_secret']
        self.api_key = self.config['api_key']
        self.model = self.config.get('model', 'spark-lite')
        self.api_url = self.config.get('api_url', 'wss://spark-api.xf-yun.com/v1.1/chat')
        self.domain = self.config.get('domain', 'lite')  # 添加domain配置
        
        # 系统提示词（角色设定）
        self.system_prompt = self._load_system_prompt()
        
        # 响应结果
        self.response_text = ""
        self.is_complete = False
        
        logger.info(f"讯飞星火API初始化完成 - APPID: {self.appid}, Model: {self.model}")
    
    def _load_system_prompt(self) -> str:
        """加载系统提示词"""
        # 尝试多个可能的路径
        possible_paths = [
            os.path.join(os.path.dirname(__file__), '讯飞星火-角色设定.txt'),
            os.path.join(os.path.dirname(os.path.dirname(__file__)), '讯飞星火-角色设定.txt'),
            '讯飞星火-角色设定.txt'
        ]
        
        for prompt_path in possible_paths:
            try:
                if os.path.exists(prompt_path):
                    with open(prompt_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        logger.info(f"成功加载角色设定文件: {prompt_path}")
                        return content
            except Exception as e:
                continue
        
        # 如果都找不到，使用内置的角色设定
        logger.warning("未找到角色设定文件，使用内置提示词")
        return """你是电学实验室助手，将用户语音指令转换为JSON格式。
元件：电阻(Ω/kΩ)、电源(V)、信号发生器(Hz/kHz)、示波器、接地
操作：添加、连接、删除、修改、运行仿真、验证KVL/KCL
术语：串联、并联、混联、节点、支路、电压、电流
输出格式：{"action":"操作","component_type":"类型","component_id":"ID","parameters":{"value":"值","unit":"单位"},"confidence":"0-1"}
示例：
"添加一千欧姆电阻R1" → {"action":"add","component_type":"resistor","component_id":"R1","parameters":{"value":"1000","unit":"Ω"},"confidence":0.95}
"连接R1和V1" → {"action":"connect","components":["R1","V1"],"confidence":0.90}
"运行仿真" → {"action":"simulate","confidence":0.98}"""
    
    def create_url(self) -> str:
        """生成鉴权URL"""
        # 解析URL
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
    
    def parse_command(self, user_input: str) -> Optional[Dict[str, Any]]:
        """
        解析用户指令
        
        Args:
            user_input: 用户输入的文本
        
        Returns:
            解析后的JSON字典，失败返回None
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
                on_open=lambda ws: self._on_open(ws, user_input)
            )
            
            # 运行WebSocket（阻塞）
            ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
            
            # 解析返回的JSON
            if self.response_text:
                return self._extract_json(self.response_text)
            
            return None
            
        except Exception as e:
            logger.error(f"讯飞星火API调用失败: {e}")
            return None
    
    def _on_open(self, ws, user_input: str):
        """WebSocket连接建立时发送请求"""
        data = {
            "header": {
                "app_id": self.appid,
                "uid": "user_001"
            },
            "parameter": {
                "chat": {
                    "domain": self.domain,  # 使用配置的domain
                    "temperature": 0.5,
                    "max_tokens": 2048
                }
            },
            "payload": {
                "message": {
                    "text": [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_input}
                    ]
                }
            }
        }
        
        ws.send(json.dumps(data))
        logger.debug(f"发送请求: {user_input}")
    
    def _on_message(self, ws, message):
        """接收响应消息"""
        try:
            data = json.loads(message)
            code = data['header']['code']
            
            if code != 0:
                logger.error(f"API返回错误: {data['header']['message']}")
                ws.close()
                return
            
            # 提取文本内容
            choices = data['payload']['choices']
            status = choices['status']
            text_list = choices['text']
            
            if len(text_list) > 0:
                text_item = text_list[0]
                # X2版本：中间消息用reasoning_content，最终消息用content
                if 'content' in text_item:
                    content = text_item['content']
                elif 'reasoning_content' in text_item:
                    content = text_item['reasoning_content']
                else:
                    content = ""
                
                self.response_text += content
            
            # 判断是否结束
            if status == 2:
                self.is_complete = True
                logger.info(f"接收完成: {self.response_text}")
                ws.close()
                
        except Exception as e:
            logger.error(f"解析响应失败: {e}")
            import traceback
            traceback.print_exc()
            ws.close()
    
    def _on_error(self, ws, error):
        """错误处理"""
        logger.error(f"WebSocket错误: {error}")
    
    def _on_close(self, ws, close_status_code, close_msg):
        """连接关闭"""
        logger.debug("WebSocket连接已关闭")
    
    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """
        从文本中提取JSON
        
        Args:
            text: 包含JSON的文本
        
        Returns:
            提取的JSON字典，失败返回None
        """
        try:
            # 尝试直接解析
            return json.loads(text)
        except json.JSONDecodeError:
            # 尝试提取JSON部分 - 查找最后一个完整的JSON对象
            import re
            # 从后往前找，因为最终的JSON通常在最后
            json_matches = list(re.finditer(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text))
            
            if json_matches:
                # 尝试从后往前解析每个匹配
                for match in reversed(json_matches):
                    try:
                        json_str = match.group()
                        result = json.loads(json_str)
                        # 验证是否包含必要的字段
                        if 'action' in result:
                            logger.info(f"成功提取JSON: {json_str}")
                            return result
                    except json.JSONDecodeError:
                        continue
            
            logger.warning(f"无法从响应中提取JSON: {text}")
            return None


# 简化的调用接口
def parse_voice_command(user_input: str, config_path: str = None) -> Optional[Dict[str, Any]]:
    """
    解析语音指令（简化接口）
    
    Args:
        user_input: 用户输入的文本
        config_path: 配置文件路径
    
    Returns:
        解析后的JSON字典
    """
    api = XFYunSparkAPI(config_path)
    return api.parse_command(user_input)


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    test_commands = [
        "添加一千欧姆电阻R1",
        "添加5V电源V1",
        "连接R1和V1",
        "运行仿真"
    ]
    
    for cmd in test_commands:
        print(f"\n测试指令: {cmd}")
        result = parse_voice_command(cmd)
        if result:
            print(f"解析结果: {json.dumps(result, ensure_ascii=False, indent=2)}")
        else:
            print("解析失败")
