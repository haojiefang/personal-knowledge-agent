import os
import sys
import json
import requests
from typing import List, Dict, Any, Optional, Union
from dotenv import load_dotenv
import time


# 加载环境变量
load_dotenv()

# 打印调试信息
print(f"Python版本: {sys.version}")

class DeepSeekClient:
    """简易DeepSeek API调用客户端 - 直接使用requests实现，不依赖OpenAI库"""
    
    def __init__(self, timeout: int = 30):
        """初始化客户端"""
        # 从环境变量获取API配置
        self.api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        self.api_base = os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")
        
        print(f"初始化DeepSeek客户端, API基地址: {self.api_base}")
        
        # 设置API请求头
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        # 默认模型参数
        self.model = "deepseek-chat"
        self.temperature = 0.7
        
        # 请求超时时间（秒）
        self.timeout = timeout
        
        # 最大重试次数
        self.max_retries = 2
    
    def chat(self, 
            messages: List[Dict[str, str]], 
            temperature: Optional[float] = None, 
            max_tokens: Optional[int] = None,
            timeout: Optional[int] = None) -> Union[str, Dict[str, Any]]:
        """
        发送聊天请求并返回回复内容
        
        Args:
            messages: 消息列表，格式为[{"role": "user", "content": "你好"}]
            temperature: 温度参数，控制随机性
            max_tokens: 最大生成token数量
            timeout: 请求超时时间（秒）
            
        Returns:
            生成的回复文本或完整响应
        """
        # 使用传入的timeout或默认值
        actual_timeout = timeout if timeout is not None else self.timeout
        
        # 初始化重试计数器
        retry_count = 0
        
        while retry_count <= self.max_retries:
            try:
                print(f"发送聊天请求, 消息数: {len(messages)}, 超时设置: {actual_timeout}秒")
                
                # 构建请求体
                payload = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature if temperature is not None else self.temperature
                }
                
                # 如果提供了max_tokens，则添加到请求中
                if max_tokens is not None:
                    payload["max_tokens"] = max_tokens
                
                # 发送请求
                endpoint = f"{self.api_base}/chat/completions"
                print(f"请求端点: {endpoint}")
                
                # 开始计时
                start_time = time.time()
                
                response = requests.post(
                    endpoint,
                    headers=self.headers,
                    json=payload,
                    timeout=actual_timeout  # 设置请求超时
                )
                
                # 计算请求耗时
                elapsed_time = time.time() - start_time
                print(f"API请求耗时: {elapsed_time:.2f}秒")
                
                # 检查响应状态
                response.raise_for_status()
                response_data = response.json()
                
                # 提取回复内容
                if "choices" in response_data and len(response_data["choices"]) > 0:
                    message = response_data["choices"][0].get("message", {})
                    return message.get("content", "")
                else:
                    return "API响应中没有找到回复内容"
                
            except requests.exceptions.Timeout:
                retry_count += 1
                wait_time = min(2 ** retry_count, 10)  # 指数退避策略
                
                if retry_count <= self.max_retries:
                    print(f"请求超时，等待{wait_time}秒后重试 ({retry_count}/{self.max_retries})")
                    time.sleep(wait_time)
                else:
                    return "错误：DeepSeek API请求超时。请检查您的网络连接或稍后再试。"
            
            except requests.exceptions.RequestException as e:
                error_message = f"API请求错误: {str(e)}"
                print(error_message)
                
                # 区分不同类型的错误
                if "401" in str(e):
                    return "错误：API密钥无效或已过期。请检查您的 DeepSeek API 密钥。"
                elif "429" in str(e):
                    return "错误：超出API使用限制。请稍后重试或检查您的使用配额。"
                elif "500" in str(e) or "503" in str(e):
                    retry_count += 1
                    if retry_count <= self.max_retries:
                        wait_time = min(2 ** retry_count, 10)
                        print(f"服务器错误，等待{wait_time}秒后重试 ({retry_count}/{self.max_retries})")
                        time.sleep(wait_time)
                    else:
                        return "错误：DeepSeek服务器暂时不可用。请稍后再试。"
                else:
                    return f"错误：{error_message}"
                    
            except Exception as e:
                print(f"未预期的错误: {e}")
                return f"错误：调用API时发生未知错误：{e}"
    
    def ask(self, prompt: str, temperature: Optional[float] = None, timeout: Optional[int] = None) -> str:
        """
        发送单个问题并获取回答（简化接口）
        
        Args:
            prompt: 提问内容
            temperature: 温度参数
            timeout: 请求超时时间（秒）
            
        Returns:
            生成的回答
        """
        # 检查输入长度，避免超出token限制
        if len(prompt) > 12000:  # 大约4000个token的粗略估计
            print(f"警告：输入文本长度为{len(prompt)}字符，可能超出模型限制，正在截断...")
            prompt = prompt[:12000] + "\n\n[注：由于内容过长，已被截断]"
        
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages, temperature, timeout=timeout) 