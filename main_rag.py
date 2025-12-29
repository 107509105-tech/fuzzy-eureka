from typing import Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

from rag_pipeline import RAGPipeline

app = FastAPI(title="RAG QA Service")
app.mount("/static", StaticFiles(directory="static"), name="static")

pipeline: RAGPipeline


@app.on_event("startup")
def startup() -> None:
    global pipeline
    pipeline = RAGPipeline()


class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = None
    doc_name: Optional[str] = None  # 文件名過濾（例如 "manual"）
    doc_type: Optional[str] = None  # 文件類型過濾（"pdf" 或 "docx"）


@app.get("/", response_class=HTMLResponse)
def serve_frontend() -> str:
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.post("/query")
def query_rag(payload: QueryRequest) -> JSONResponse:
    question = payload.query.strip()
    if not question:
        return JSONResponse({"status": "error", "error": "請輸入查詢內容"}, status_code=400)

    try:
        # 構建過濾表達式
        filter_expr = None
        filters = []

        if payload.doc_name:
            filters.append(f'doc_name == "{payload.doc_name}"')
        if payload.doc_type:
            filters.append(f'doc_type == "{payload.doc_type}"')

        if filters:
            filter_expr = " && ".join(filters)

        result = pipeline.query(question, top_k=payload.top_k, doc_filter=filter_expr)
        return JSONResponse(result)
    except Exception as exc:  # 捕捉後端錯誤，避免洩漏堆疊
        return JSONResponse({"status": "error", "error": str(exc)}, status_code=500)


@app.get("/documents")
def get_documents() -> JSONResponse:
    """獲取所有可用的文件列表"""
    try:
        # 載入 collection
        pipeline.vector_store.collection.load()

        # 使用 query 來獲取所有不同的文件
        # 這裡我們使用一個簡單的方法：搜尋一個隨機向量並獲取結果
        import numpy as np
        from config import VECTOR_DIM

        # 隨機向量搜尋來獲取文件信息
        random_vector = np.random.rand(VECTOR_DIM).astype(np.float32)
        results = pipeline.vector_store.search(random_vector, top_k=100)

        # 提取唯一的文件名和類型
        docs_set = set()
        for result in results:
            doc_name = result.get('doc_name', '')
            doc_type = result.get('doc_type', 'pdf')
            if doc_name:
                docs_set.add((doc_name, doc_type))

        # 轉換為列表並排序
        documents = [{"name": name, "type": dtype} for name, dtype in sorted(docs_set)]

        return JSONResponse({"status": "success", "documents": documents})
    except Exception as exc:
        return JSONResponse({"status": "error", "error": str(exc)}, status_code=500)


if __name__ == "__main__":
    uvicorn.run("main_rag:app", host="0.0.0.0", port=8002, reload=True)
