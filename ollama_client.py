"""
Ollama API客户端
用于连接本地Ollama服务进行模型调用
"""

import os
import sys
import json
import requests
import time
from typing import List, Dict, Any, Optional, Union

class OllamaClient:
    """简易Ollama API客户端，调用本地Ollama服务"""
    
    def __init__(self, api_base: str = "http://localhost:11434", timeout: int = 30):
        """初始化客户端
        
        Args:
            api_base: Ollama API基础URL，默认为http://localhost:11434
            timeout: 请求超时时间（秒），默认30秒
        """
        self.api_base = api_base
        self.timeout = timeout
        self.max_retries = 2
        self.model = "llama3"  # 默认模型
        
        print(f"初始化Ollama客户端，API地址: {self.api_base}")
        
        # 检查Ollama服务是否可用
        self._check_availability()
    
    def _check_availability(self):
        """检查Ollama服务是否可用并列出可用模型"""
        try:
            # 尝试获取可用模型列表
            response = requests.get(f"{self.api_base}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                if models:
                    model_names = [m.get("name") for m in models]
                    print(f"Ollama服务可用，检测到 {len(model_names)} 个模型: {', '.join(model_names)}")
                else:
                    print("Ollama服务可用，但未检测到已下载的模型")
            else:
                print(f"Ollama服务响应异常，状态码: {response.status_code}")
        except Exception as e:
            print(f"检查Ollama服务可用性时出错: {e}")
            print("Ollama服务可能未运行，请确保已安装并启动Ollama服务")
    
    def get_available_models(self) -> List[str]:
        """获取可用的Ollama模型列表"""
        try:
            response = requests.get(f"{self.api_base}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                return [m.get("name") for m in models]
            return []
        except:
            return []
    
    def set_model(self, model_name: str):
        """设置默认使用的模型名称"""
        self.model = model_name
        print(f"Ollama客户端已设置模型: {model_name}")
    
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
        # 使用传入的timeout或默认值
        actual_timeout = timeout if timeout is not None else self.timeout
        
        # 初始化重试计数器
        retry_count = 0
        
        # 检查输入长度，避免过长
        if len(prompt) > 12000:
            print(f"警告：输入文本长度为{len(prompt)}字符，可能过长，正在截断...")
            prompt = prompt[:12000] + "\n\n[注：由于内容过长，已被截断]"
        
        while retry_count <= self.max_retries:
            try:
                print(f"使用generate接口请求模型: {self.model}")
                
                # 构建请求
                payload = {
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False
                }
                
                # 添加可选参数
                if temperature is not None:
                    payload["temperature"] = temperature
                
                # 发送请求
                endpoint = f"{self.api_base}/api/generate"
                print(f"请求端点: {endpoint}")
                
                # 开始计时
                start_time = time.time()
                
                response = requests.post(
                    endpoint,
                    json=payload,
                    timeout=actual_timeout
                )
                
                # 计算请求耗时
                elapsed_time = time.time() - start_time
                print(f"Ollama请求耗时: {elapsed_time:.2f}秒")
                
                # 检查响应状态
                response.raise_for_status()
                response_data = response.json()
                
                # 提取回复内容
                if "response" in response_data:
                    answer = response_data["response"].strip()
                    print(f"成功获取Ollama回复，长度: {len(answer)}字符")
                    return answer
                else:
                    print(f"警告: 响应中没有'response'字段: {list(response_data.keys())}")
                    return "模型响应中未找到有效内容"
            
            except requests.exceptions.Timeout:
                retry_count += 1
                wait_time = min(2 ** retry_count, 10)  # 指数退避策略
                
                if retry_count <= self.max_retries:
                    print(f"请求超时，等待{wait_time}秒后重试 ({retry_count}/{self.max_retries})")
                    time.sleep(wait_time)
                else:
                    return "错误：Ollama请求超时，本地服务可能响应缓慢或未启动"
            
            except requests.exceptions.ConnectionError:
                return "错误：无法连接到Ollama服务，请确保Ollama已启动"
                
            except requests.exceptions.RequestException as e:
                error_message = f"Ollama请求错误: {str(e)}"
                print(error_message)
                
                if "404" in str(e):
                    return f"错误：找不到指定的模型 '{self.model}'，请确保已下载该模型"
                elif "500" in str(e):
                    retry_count += 1
                    if retry_count <= self.max_retries:
                        wait_time = min(2 ** retry_count, 10)
                        print(f"服务器错误，等待{wait_time}秒后重试 ({retry_count}/{self.max_retries})")
                        time.sleep(wait_time)
                    else:
                        return "错误：Ollama服务器内部错误"
                else:
                    return f"错误：{error_message}"
                    
            except Exception as e:
                print(f"未预期的错误: {e}")
                return f"错误：调用Ollama时发生未知错误：{e}"
    
    def chat(self, 
            messages: List[Dict[str, str]], 
            temperature: Optional[float] = None, 
            max_tokens: Optional[int] = None,
            timeout: Optional[int] = None) -> Union[str, Dict[str, Any]]:
        """
        发送聊天请求并返回回复内容 - 简化为使用ask方法
        """
        # 获取最后一条用户消息作为问题
        last_user_message = None
        for msg in reversed(messages):
            if msg["role"] == "user":
                last_user_message = msg["content"]
                break
        
        if not last_user_message:
            return "错误：未找到用户消息"
        
        # 调用ask方法
        return self.ask(last_user_message, temperature, timeout) 