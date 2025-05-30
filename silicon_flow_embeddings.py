import os
import requests
from typing import List, Optional
from dotenv import load_dotenv
from langchain.embeddings.base import Embeddings

# 加载环境变量
load_dotenv()

class SiliconFlowEmbeddings(Embeddings):
    """硅基流动API嵌入类，实现LangChain的Embeddings接口"""
    
    def __init__(self, model_name: str = "BAAI/bge-m3", api_key: str = None, batch_size: int = 16):
        """
        初始化硅基流动嵌入客户端
        
        Args:
            model_name: 使用的嵌入模型名称
            api_key: 硅基流动API密钥，若未提供则从环境变量读取
            batch_size: 批处理大小，控制每次API请求的文本数量
        """
        self.model_name = model_name
        self.api_key = api_key or os.getenv("SILICON_FLOW_API_KEY")
        self.batch_size = batch_size
        if not self.api_key:
            raise ValueError("Silicon Flow API密钥未设置，请在.env文件中添加SILICON_FLOW_API_KEY")
        self.url = "https://api.siliconflow.cn/v1/embeddings"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json; charset=utf-8"
        }
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        为多个文本生成嵌入向量
        
        Args:
            texts: 需要嵌入的文本列表
            
        Returns:
            嵌入向量列表，每个文本对应一个向量
        """
        # 批处理，避免请求体过大
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i+self.batch_size]
            try:
                print(f"处理第 {i//self.batch_size + 1} 批文本，共 {len(batch_texts)} 条")
                response = requests.post(
                    self.url,
                    headers=self.headers,
                    json={
                        "model": self.model_name,
                        "input": batch_texts
                    }
                )
                response.raise_for_status()
                data = response.json()
                batch_embeddings = [embedding["embedding"] for embedding in data["data"]]
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                print(f"获取嵌入向量失败: {e}")
                # 出错时返回空向量
                empty_embeddings = [[0.0] * 1024] * len(batch_texts)
                all_embeddings.extend(empty_embeddings)
        
        return all_embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """
        为单个查询文本生成嵌入向量
        
        Args:
            text: 需要嵌入的查询文本
            
        Returns:
            查询文本的嵌入向量
        """
        embeddings = self.embed_documents([text])
        return embeddings[0] if embeddings else [0.0] * 1024 