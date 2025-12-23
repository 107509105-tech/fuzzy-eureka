# PDF RAG 問答系統

基於 Gemma 3 和 BGE-M3 的智能 PDF 文件問答系統。支援簡體中文 PDF 輸入，並使用繁體中文回答問題。

## 功能特色

- **智能摘要生成**：使用 Gemma 3 Vision 模型為每個 PDF 頁面生成結構化摘要
- **語義搜尋**：使用 BGE-M3 向量模型進行高精度的語義檢索
- **本地向量資料庫**：使用 Milvus Lite，無需額外安裝 Docker
- **Web 介面**：友善的 FastAPI Web 應用，支援拖放上傳和即時問答
- **繁簡中文支援**：輸入簡體中文 PDF，用繁體中文回答問題

## 技術棧

- **LLM**: Gemma 3 (透過 OpenAI API 格式)
- **Embedding**: BGE-M3 (BAAI/bge-m3)
- **向量資料庫**: Milvus Lite
- **Web 框架**: FastAPI + Jinja2
- **PDF 處理**: pdf2image + Poppler

## 安裝步驟

### 1. 系統依賴

**macOS**:
```bash
brew install poppler
```

**Ubuntu/Debian**:
```bash
sudo apt-get install poppler-utils
```

**Windows**:
- 下載並安裝 Poppler: https://github.com/oschwartz10612/poppler-windows/releases

### 2. Python 環境

建議使用 Python 3.9 或以上版本。

```bash
# 建立虛擬環境（推薦）
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate  # Windows

# 安裝依賴
pip install -r requirements.txt
```

### 3. 配置設定

編輯 `config.py` 檔案，設定您的 API 憑證：

```python
# LLM API 設定
LLM_API_KEY = "your-api-key-here"  # 替換為您的 API key
LLM_API_BASE = "https://api.example.com/v1"  # 替換為您的 API 端點
LLM_MODEL_NAME = "gemma-3"  # 模型名稱
```

或使用環境變數：

```bash
export LLM_API_KEY="your-api-key-here"
export LLM_API_BASE="https://api.example.com/v1"
export LLM_MODEL_NAME="gemma-3"
```

## 使用方式

### 方法 1: Web 介面（推薦）

1. 啟動 FastAPI 應用：

```bash
python main.py
```

或使用 uvicorn：

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

2. 在瀏覽器中打開：http://localhost:8000

3. 使用介面：
   - 上傳 PDF 檔案（拖放或點擊選擇）
   - 等待索引完成
   - 在問答區輸入問題
   - 查看答案和來源頁面

### 方法 2: 命令列介面

**索引 PDF**：

```bash
python rag_pipeline.py index path/to/your.pdf
```

**查詢問答**：

```bash
python rag_pipeline.py query "你的問題"
```

**查看統計**：

```bash
python rag_pipeline.py stats
```

## 專案結構

```
manual_rag/
├── config.py              # 配置文件
├── requirements.txt       # Python 依賴
├── document_processor.py  # PDF 處理模組
├── summary_generator.py   # 摘要生成模組
├── embedder.py           # 向量化模組
├── vector_store.py       # Milvus 存儲模組
├── rag_pipeline.py       # RAG 主流程
├── main.py               # FastAPI 應用
├── static/               # 靜態資源
│   └── style.css        # CSS 樣式
├── templates/            # HTML 模板
│   └── index.html       # 主頁面
├── temp_images/          # 暫存 PDF 圖像
├── pdfs/                 # 上傳的 PDF 存放處
└── milvus_rag.db        # Milvus Lite 資料庫（自動生成）
```

## API 端點

- `GET /` - 主頁面
- `POST /upload` - 上傳並索引 PDF
- `POST /query` - 提交問題查詢
- `GET /image/{filename}` - 取得暫存圖像
- `GET /stats` - 系統統計資訊
- `DELETE /reset` - 重置資料庫
- `GET /health` - 健康檢查

## 測試各模組

各模組都可以獨立測試：

**測試 PDF 處理**：
```bash
python document_processor.py path/to/test.pdf
```

**測試向量嵌入**：
```bash
python embedder.py
```

**測試向量存儲**：
```bash
python vector_store.py
```

**測試摘要生成**：
```bash
python summary_generator.py path/to/image.png
```

## 注意事項

1. **首次運行**：第一次運行會自動下載 BGE-M3 模型（約 2GB），請確保網路連接穩定

2. **API 配置**：確保在 `config.py` 中正確設定 API key 和端點

3. **記憶體需求**：BGE-M3 模型需要約 4GB 記憶體

4. **圖像暫存**：PDF 轉換的圖像會暫存在 `temp_images/` 目錄，可定期清理

5. **資料庫檔案**：Milvus Lite 資料存在本地檔案 `milvus_rag.db`，刪除即可重置資料庫

## 常見問題

**Q: 無法轉換 PDF？**
A: 請確認已安裝 Poppler。macOS 使用 `brew install poppler`

**Q: 模型下載失敗？**
A: 首次運行會下載 BGE-M3 模型，請確保網路連接和 Hugging Face 訪問正常

**Q: API 調用失敗？**
A: 檢查 `config.py` 中的 API key 和 base URL 是否正確

**Q: 向量資料庫錯誤？**
A: 嘗試刪除 `milvus_rag.db` 檔案重新初始化

## 授權

本專案僅供學習和研究使用。

## 貢獻

歡迎提交 Issue 和 Pull Request！
