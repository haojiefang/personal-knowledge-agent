"""
知识库助手全局配置文件
包含各种API密钥和服务设置
"""

import os
from dotenv import load_dotenv

# 加载.env文件中的环境变量(如果存在)
load_dotenv()

# 语言模型服务配置
LLM_SERVICES = {
    # DeepSeek配置
    "deepseek": {
        "name": "DeepSeek",
        "api_key": os.environ.get("DEEPSEEK_API_KEY", ""),
        "api_base": os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1"),
        "model": "deepseek-chat",
        "enabled": bool(os.environ.get("DEEPSEEK_API_KEY", "")),
        "description": "DeepSeek云端大模型"
    },
    # Ollama本地服务配置
    "ollama": {
        "name": "Ollama本地模型",
        "api_base": os.environ.get("OLLAMA_API_BASE", "http://localhost:11434"),
        "models": [
            "qwen2.5:latest", 
            "deepseek-r1:8b", 
            "deepseek-r1:1.5b"# 保留原有模型，但会动态检测
        ],
        "default_model": "qwen2.5:latest",  # 更新为实际可用的模型
        "enabled": os.environ.get("OLLAMA_ENABLED", "false").lower() == "true",  # 通过环境变量控制，默认禁用
        "description": "本地Ollama服务 (需先安装)"
    }
}

# 嵌入模型配置
EMBEDDING_SERVICE = {
    "name": "Silicon Flow",
    "api_key": os.environ.get("SILICON_FLOW_API_KEY", ""),
    "api_base": os.environ.get("SILICON_FLOW_API_BASE", "https://api.siliconflow.cn/v1"),
    "model": "BAAI/bge-m3"
}

# 应用设置
APP_SETTINGS = {
    "max_context_length": 8000,  # 上下文最大字符数
    "max_tokens": 4000,          # 生成最大token数
    "temperature": 0.3,          # 生成温度
    "request_timeout": 45,       # 请求超时时间(秒)
    "max_retries": 2             # 最大重试次数
} 