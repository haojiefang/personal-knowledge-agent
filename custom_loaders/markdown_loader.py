
from typing import List
from langchain.docstore.document import Document

class SimpleMarkdownLoader:
    """简化版Markdown文件加载器"""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
    
    def load(self) -> List[Document]:
        """加载Markdown文件内容"""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # 简单清理Markdown标记
            lines = text.split('\n')
            cleaned_lines = []
            for line in lines:
                # 移除标题符号
                if line.startswith('#'):
                    line = line.lstrip('#').strip()
                # 移除链接
                while '[' in line and '](' in line and ')' in line:
                    start = line.find('[')
                    middle = line.find('](', start)
                    end = line.find(')', middle)
                    if start < middle < end:
                        link_text = line[start+1:middle]
                        line = line[:start] + link_text + line[end+1:]
                    else:
                        break
                # 移除粗体和斜体
                line = line.replace('**', '').replace('__', '').replace('*', '').replace('_', '')
                cleaned_lines.append(line)
            
            cleaned_text = '\n'.join(cleaned_lines)
            metadata = {"source": self.file_path}
            return [Document(page_content=cleaned_text, metadata=metadata)]
        except Exception as e:
            print(f"加载Markdown文件时出错: {e}")
            return []
