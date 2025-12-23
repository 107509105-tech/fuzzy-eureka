import os
from pathlib import Path

# LLM Configuration (Gemma 3 via OpenAI API format)
LLM_API_KEY = os.getenv("LLM_API_KEY", "your-api-key-here")
LLM_API_BASE = os.getenv("LLM_API_BASE", "https://api.example.com/v1")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gemma-3")

# Embedding Model Configuration
BGE_MODEL_NAME = "BAAI/bge-m3"
VECTOR_DIM = 1024  # BGE-M3 output dimension

# Milvus Configuration
MILVUS_URI = "./milvus_rag.db"  # Milvus Lite local file
COLLECTION_NAME = "pdf_rag_collection"

# RAG Configuration
TOP_K = 3  # Number of documents to retrieve
MAX_RETRIES = 3  # Max retries for API calls

# Paths
BASE_DIR = Path(__file__).parent
TEMP_IMAGES_DIR = BASE_DIR / "temp_images"
PDFS_DIR = BASE_DIR / "pdfs"
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

# PDF Processing
PDF_DPI = 300  # Resolution for PDF to image conversion

# Prompts (from project.md)
SUMMARY_PROMPT = """你是一個專業的文件分析助手。請仔細閱讀這個文件頁面，並提供完整的結構化摘要。

請按以下格式輸出：

【頁面主題】
簡述這一頁的主要內容主題

【詳細內容】
- 列出所有重要資訊、數據、定義
- 如有表格，請完整轉錄表格內容（用 markdown 表格格式）
- 如有流程圖或圖表，請描述其內容和關係

【關鍵術語】
列出頁面中出現的專業術語或關鍵字（用逗號分隔）

【備註】
任何需要特別注意的事項"""

QUERY_PROMPT_TEMPLATE = """你是一個專業的文件問答助手。根據提供的文件頁面圖像，回答使用者的問題。

使用者問題：{query}

請注意：
- 只根據圖像中的內容回答，不要編造資訊
- 如果圖像中沒有相關資訊，請明確說明
- 如果答案涉及表格或數據，請準確引用
- 用繁體中文回答"""
