# 个人知识代理 (Personal Knowledge Agent)

一个基于 RAG (检索增强生成) 技术的智能知识库助手，支持多种文档格式，提供精准的问答服务。

## 🚀 项目特色

- **多格式文档支持**: PDF、Word、Markdown、文本文件
- **智能分词与检索**: 基于 LangChain 的递归文本分割器
- **高性能向量存储**: 使用 ChromaDB 持久化存储
- **多模型支持**: 集成 DeepSeek、Ollama 等大语言模型
- **复合问题处理**: 自动识别并拆分复合问题
- **严格过滤模式**: 精确文档查询与相似度匹配
- **Web 界面**: 基于 Streamlit 的直观用户界面

## 📋 系统要求

- Python 3.8+
- 8GB+ RAM (推荐)
- 可选: NVIDIA GPU (用于本地 Ollama 模型)

## 🛠 安装配置

### 1. 克隆项目

```bash
用git克隆或下载项目到本地
```

### 2. 创建虚拟环境

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements_full.txt
```

### 4. 环境配置

创建 `.env` 文件，配置 API 密钥：

```env
# 必需: Silicon Flow 嵌入模型 API
SILICON_FLOW_API_KEY=your_silicon_flow_api_key
SILICON_FLOW_API_BASE=https://api.siliconflow.cn/v1

# DeepSeek 大语言模型 (可选)
DEEPSEEK_API_KEY=your_deepseek_api_key
DEEPSEEK_API_BASE=https://api.deepseek.com/v1

# Ollama 本地模型 (可选)
OLLAMA_ENABLED=true
OLLAMA_API_BASE=http://localhost:11434
```

## 🚀 快速开始

### 启动应用

```bash
streamlit run app.py
```

应用将在 `http://localhost:8501` 启动。

### 基本使用流程

1. **上传文档**: 在侧边栏选择并上传支持的文档格式
2. **处理文档**: 点击"处理文档"按钮，系统自动分词并构建向量索引
3. **智能问答**: 在主界面输入问题，获取基于文档内容的回答
4. **文档筛选**: 选择特定文档进行精确查询
5. **知识库管理**: 在管理界面查看和维护知识库

## 🔍 RAG 技术详解

### 分词策略 (Text Splitting)

项目采用 LangChain 的 `RecursiveCharacterTextSplitter`，实现智能分词：

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,        # 每个文本块大小
    chunk_overlap=50,      # 块间重叠字符数
    separators=["\n\n", "\n", "。", "！", "？", ".", " ", ""]  # 分割优先级
)
```

**分词特点**:

- **递归分割**: 按分隔符优先级递归分割，保持语义完整性
- **智能重叠**: 50字符重叠确保上下文连续性
- **中文优化**: 针对中文标点符号优化分割逻辑
- **元数据保持**: 每个文本块保留完整的源文档信息

### 检索机制 (Retrieval)

#### 向量化存储

- **嵌入模型**: Silicon Flow BGE-M3 模型
- **向量数据库**: ChromaDB 持久化存储
- **相似度计算**: 余弦相似度匹配

#### 检索流程

```python
# 1. 问题向量化
query_embedding = embedding_function(question)

# 2. 相似度检索
results = collection.query(
    query_texts=[question],
    n_results=5,           # 返回最相关的5个文档块
    include=["documents", "metadatas"]
)

# 3. 文档过滤 (可选)
if document_filters:
    # 精确文档匹配或模糊匹配
    filtered_results = apply_document_filters(results, filters)
```

#### 检索模式

- **全局检索**: 在所有文档中搜索相关内容
- **文档过滤**: 限定特定文档范围内检索
- **严格模式**: 仅在选定文档中精确匹配
- **相似匹配**: 基于语义相似度的模糊匹配

### 复合问题处理

系统自动识别包含多个问号或分号的复合问题：

```python
def split_compound_question(question):
    # 检测多个问号或分号
    delimiters = ['？', '?', '；', ';']
    if count_delimiters(question) > 1:
        # 拆分为子问题
        sub_questions = split_by_delimiters(question)
        # 分别检索，合并结果
        return sub_questions
    return [question]
```

### 上下文构建

智能上下文长度管理，确保 LLM 生成质量：

```python
MAX_CONTEXT_LENGTH = 8000  # 配置的上下文限制

# 按文档长度排序，优先保留短文档
contexts.sort(key=lambda x: len(x))

# 动态添加上下文直到达到长度限制
for context in contexts:
    if total_length + len(context) <= MAX_CONTEXT_LENGTH:
        final_contexts.append(context)
        total_length += len(context)
```

## 🎯 优化建议

### 1. 分词优化

#### 当前实现

```python
chunk_size=500, chunk_overlap=50
```

#### 优化方案

```python
# 根据文档类型动态调整
def get_optimal_chunk_config(file_type):
    configs = {
        'pdf': {'chunk_size': 800, 'chunk_overlap': 80},    # 学术文档
        'md': {'chunk_size': 600, 'chunk_overlap': 60},     # 技术文档  
        'txt': {'chunk_size': 400, 'chunk_overlap': 40},    # 一般文本
        'docx': {'chunk_size': 700, 'chunk_overlap': 70}    # 商务文档
    }
    return configs.get(file_type, {'chunk_size': 500, 'chunk_overlap': 50})
```

#### 语义分割增强

```python
# 集成句子级别的语义分割
from sentence_transformers import SentenceTransformer

def semantic_chunking(text, max_chunk_size=500):
    sentences = split_into_sentences(text)
    chunks = []
    current_chunk = []
    current_size = 0
  
    for sentence in sentences:
        if current_size + len(sentence) > max_chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_size = len(sentence)
        else:
            current_chunk.append(sentence)
            current_size += len(sentence)
  
    if current_chunk:
        chunks.append(' '.join(current_chunk))
  
    return chunks
```

### 2. 检索精度优化

#### 混合检索策略

```python
def hybrid_retrieval(question, collection, alpha=0.7):
    # 向量检索
    vector_results = collection.query(query_texts=[question], n_results=10)
  
    # BM25 关键词检索
    bm25_results = bm25_search(question, collection)
  
    # 结果融合 (RRF - Reciprocal Rank Fusion)
    final_results = reciprocal_rank_fusion(
        vector_results, bm25_results, alpha=alpha
    )
    return final_results
```

#### 重排序机制

```python
def rerank_results(question, results, rerank_model):
    # 使用专门的重排序模型
    pairs = [(question, doc) for doc in results['documents'][0]]
    scores = rerank_model.compute_score(pairs)
  
    # 按重排序分数重新排列
    sorted_indices = sorted(range(len(scores)), 
                          key=lambda i: scores[i], reverse=True)
  
    return reorder_results(results, sorted_indices)
```

### 3. 性能优化

#### 向量索引优化

```python
# ChromaDB 配置优化
collection = client.create_collection(
    name="documents",
    embedding_function=embedding_function,
    metadata={
        "hnsw:space": "cosine",           # 余弦相似度
        "hnsw:construction_ef": 200,      # 构建时邻居数
        "hnsw:M": 16,                     # 连接数
        "hnsw:search_ef": 100             # 搜索时邻居数
    }
)
```

#### 缓存机制

```python
# 查询结果缓存
@lru_cache(maxsize=1000)
def cached_query(question_hash, doc_filters_hash):
    return collection.query(...)

# 嵌入向量缓存
class CachedEmbeddingFunction:
    def __init__(self, base_function):
        self.base_function = base_function
        self.cache = {}
  
    def __call__(self, texts):
        cached_results = []
        new_texts = []
      
        for text in texts:
            text_hash = hash(text)
            if text_hash in self.cache:
                cached_results.append(self.cache[text_hash])
            else:
                new_texts.append(text)
      
        if new_texts:
            new_embeddings = self.base_function(new_texts)
            for text, embedding in zip(new_texts, new_embeddings):
                self.cache[hash(text)] = embedding
                cached_results.append(embedding)
      
        return cached_results
```

### 4. 内容质量优化

#### 文档预处理增强

```python
def enhanced_document_preprocessing(file_path, file_type):
    if file_type == 'pdf':
        # OCR 处理扫描版PDF
        if is_scanned_pdf(file_path):
            text = ocr_pdf(file_path)
        else:
            text = extract_pdf_text(file_path)
      
        # 清理PDF特有噪声
        text = clean_pdf_artifacts(text)
      
    elif file_type == 'docx':
        # 保留格式信息
        text, formatting = extract_docx_with_formatting(file_path)
      
    # 通用文本清理
    text = clean_common_artifacts(text)
  
    return text
```

#### 智能元数据提取

```python
def extract_smart_metadata(document_text, file_path):
    metadata = {
        'source': file_path,
        'filename': os.path.basename(file_path),
        'file_type': get_file_extension(file_path),
        'timestamp': int(time.time()),
      
        # 内容分析
        'language': detect_language(document_text),
        'topics': extract_topics(document_text),
        'entities': extract_entities(document_text),
        'word_count': len(document_text.split()),
        'reading_time': estimate_reading_time(document_text)
    }
    return metadata
```

### 5. 多模态扩展

#### 图像内容理解

```python
# 为包含图像的文档添加视觉理解
def process_document_with_images(file_path):
    if file_type == 'pdf':
        images = extract_images_from_pdf(file_path)
        for image in images:
            # 使用多模态模型理解图像
            image_description = vision_model.describe(image)
            # 将图像描述作为文档内容的一部分
            document_text += f"\n[图像描述: {image_description}]"
  
    return document_text
```

### 6. 用户体验优化

#### 智能问题建议

```python
def suggest_questions(documents):
    # 基于文档内容生成推荐问题
    questions = []
    for doc in documents:
        # 提取关键主题
        topics = extract_key_topics(doc.content)
        # 生成相关问题
        for topic in topics:
            suggested = generate_questions_for_topic(topic)
            questions.extend(suggested)
  
    return deduplicate_questions(questions)
```

#### 对话历史管理

```python
class ConversationManager:
    def __init__(self):
        self.history = []
        self.context_memory = {}
  
    def add_interaction(self, question, answer, context_docs):
        self.history.append({
            'question': question,
            'answer': answer,
            'context_docs': context_docs,
            'timestamp': time.time()
        })
  
    def get_relevant_history(self, current_question):
        # 返回与当前问题相关的历史对话
        return semantic_search_history(current_question, self.history)
```

## 🔧 高级配置

### 模型配置优化

编辑 `config.py` 文件进行高级配置：

```python
# 性能调优
APP_SETTINGS = {
    "max_context_length": 12000,    # 增加上下文长度
    "max_tokens": 6000,             # 增加生成长度
    "temperature": 0.1,             # 降低随机性
    "request_timeout": 60,          # 增加超时时间
    "max_retries": 3,               # 增加重试次数
  
    # 检索配置
    "retrieval_top_k": 8,           # 检索文档数量
    "rerank_top_k": 5,              # 重排序后保留数量
    "similarity_threshold": 0.7,     # 相似度阈值
}
```

## 🚨 故障排除

### 常见问题

1. **知识库损坏**

   ```bash
   # 在应用中点击"强制重建知识库"
   # 或手动删除数据库目录
   rm -rf ./chroma_db
   ```
2. **内存不足**

   - 减少 `chunk_size` 参数
   - 限制同时处理的文档数量
   - 增加系统内存
3. **API 调用失败**

   - 检查网络连接
   - 验证 API 密钥有效性
   - 查看 API 余额和调用限制

## 🤝 贡献指南

1. Fork 项目
2. 创建特性分支: `git checkout -b feature/new-feature`
3. 提交更改: `git commit -m 'Add new feature'`
4. 推送分支: `git push origin feature/new-feature`
5. 提交 Pull Request

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

## 👨‍💻 作者

**haojiefang** - 项目创建者和主要维护者

---

## 🙏 致谢

- [LangChain](https://github.com/langchain-ai/langchain) - 强大的 LLM 应用框架
- [ChromaDB](https://github.com/chroma-core/chroma) - 高性能向量数据库
- [Streamlit](https://github.com/streamlit/streamlit) - 优雅的 Web 应用框架
- [Silicon Flow](https://siliconflow.cn/) - 高质量嵌入模型服务

如有问题或建议，欢迎提交 Issue 或 Pull Request！
