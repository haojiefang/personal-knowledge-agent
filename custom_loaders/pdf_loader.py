
import pypdf
from typing import List, Optional
from langchain.docstore.document import Document

class SimplePDFLoader:
    """简化版PDF加载器，避免依赖langchain_community中的问题模块"""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
    
    def load(self) -> List[Document]:
        """加载PDF文件内容"""
        try:
            pdf = pypdf.PdfReader(self.file_path)
            documents = []
            
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text.strip():  # 忽略空页
                    metadata = {
                        "source": self.file_path,
                        "page": i + 1,
                        "total_pages": len(pdf.pages)
                    }
                    documents.append(Document(page_content=text, metadata=metadata))
            
            return documents
        except Exception as e:
            print(f"加载PDF时出错: {e}")
            return []
