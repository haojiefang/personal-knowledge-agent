
from typing import List
from langchain.docstore.document import Document

class SimpleTextLoader:
    """简化版文本文件加载器"""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
    
    def load(self) -> List[Document]:
        """加载文本文件内容"""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            metadata = {"source": self.file_path}
            return [Document(page_content=text, metadata=metadata)]
        except Exception as e:
            print(f"加载文本文件时出错: {e}")
            return []
