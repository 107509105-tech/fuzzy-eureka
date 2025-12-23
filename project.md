## 專案目標
建立一個 PDF RAG 系統，使用 Gemma 3 生成頁面摘要，BGE-M3 做向量嵌入，Milvus 作為向量資料庫。
輸入一個簡體中文PDF，用繁體中文回答
## 技術棧
- Gemma 3 
- BGE-M3 embedding 模型
- Milvus 向量資料庫
- pdf2image + poppler 處理 PDF
- fastapi + HTML CSS簡易前端
## 系統架構

### 1. 文件處理模組 (document_processor.py)
- PDF 轉圖像（每頁一張）
- 圖像暫存管理
- 支援批次處理多個 PDF

### 2. 摘要生成模組 (summary_generator.py)
- 呼叫 Gemma 3 API
- 每頁圖像生成結構化摘要
- 摘要 prompt 設計（見下方）
- 錯誤重試機制

### 3. 向量化模組 (embedder.py)
- 載入 BGE-M3 模型
- 批次生成 embedding
- 支援繁體中文簡體中文混合內容

### 4. Milvus 存儲模組 (vector_store.py)
- 連接 Milvus（支援 Milvus Lite 或遠端）
- Collection schema 設計：
  - id: INT64 (primary key)
  - embedding: FLOAT_VECTOR (dim=1024, BGE-M3)
  - text_summary: VARCHAR (摘要文字)
  - page_num: INT32
  - doc_name: VARCHAR
  - image_base64: VARCHAR (或存檔案路徑)
- 建立 HNSW 索引
- 實作 insert / search 方法

### 5. 檢索問答模組 (rag_pipeline.py)
- 接收使用者問題
- BGE-M3 編碼查詢
- Milvus 相似度搜尋（top_k=3）
- 組合檢索結果 + 原始頁面圖像
- 呼叫 Gemma 3 生成最終回答

### 6. 主程式 (main.py)
- CLI 介面
- 兩個模式：
  - `index`: 建立索引 (python main.py index --pdf_dir ./pdfs)
  - `query`: 查詢問答 (python main.py query --question "...")

## Gemma 3 Vision Prompt 設計

### 建檔時的摘要 Prompt：
你是一個專業的文件分析助手。請仔細閱讀這個文件頁面，並提供完整的結構化摘要。
請按以下格式輸出：
【頁面主題】
簡述這一頁的主要內容主題
【詳細內容】

列出所有重要資訊、數據、定義
如有表格，請完整轉錄表格內容（用 markdown 表格格式）
如有流程圖或圖表，請描述其內容和關係

【關鍵術語】
列出頁面中出現的專業術語或關鍵字（用逗號分隔）
【備註】
任何需要特別注意的事項

### 查詢時的回答 Prompt：
你是一個專業的文件問答助手。根據提供的文件頁面圖像，回答使用者的問題。
使用者問題：{query}
請注意：

只根據圖像中的內容回答，不要編造資訊
如果圖像中沒有相關資訊，請明確說明
如果答案涉及表格或數據，請準確引用
用繁體中文回答