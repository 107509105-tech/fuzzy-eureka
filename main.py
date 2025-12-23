"""
FastAPI Web 應用主程式
提供 PDF 上傳、索引和問答的 Web 介面
"""
import os
import shutil
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from pydantic import BaseModel
from rag_pipeline import RAGPipeline
import config

# 初始化 FastAPI 應用
app = FastAPI(title="PDF RAG 系統", description="基於 Gemma 3 和 BGE-M3 的 PDF 問答系統")

# 掛載靜態文件和模板
app.mount("/static", StaticFiles(directory=str(config.STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(config.TEMPLATES_DIR))

# 初始化 RAG pipeline（全域變數，應用啟動時初始化）
rag_pipeline: Optional[RAGPipeline] = None


@app.on_event("startup")
async def startup_event():
    """應用啟動時初始化 RAG pipeline"""
    global rag_pipeline
    print("\n" + "="*60)
    print("啟動 PDF RAG 系統...")
    print("="*60 + "\n")

    # 確保必要目錄存在
    config.TEMP_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    config.PDFS_DIR.mkdir(parents=True, exist_ok=True)

    # 初始化 RAG pipeline
    rag_pipeline = RAGPipeline()

    print("\n" + "="*60)
    print("系統啟動完成！")
    print("="*60 + "\n")


# Pydantic 模型
class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 3


class QueryResponse(BaseModel):
    status: str
    answer: str
    sources: list
    question: str


class UploadResponse(BaseModel):
    status: str
    message: str
    pages_processed: Optional[int] = None
    total_docs: Optional[int] = None


# API 端點

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """首頁"""
    stats = rag_pipeline.get_stats() if rag_pipeline else {}
    return templates.TemplateResponse("index.html", {
        "request": request,
        "stats": stats
    })


@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """
    上傳 PDF 並建立索引

    Args:
        file: 上傳的 PDF 檔案

    Returns:
        上傳和索引結果
    """
    if not rag_pipeline:
        raise HTTPException(status_code=500, detail="RAG pipeline 未初始化")

    # 檢查檔案類型
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="只支援 PDF 檔案")

    try:
        # 儲存上傳的檔案
        pdf_path = config.PDFS_DIR / file.filename
        with open(pdf_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print(f"\n收到 PDF 上傳: {file.filename}")

        # 索引 PDF
        result = rag_pipeline.index_pdf(str(pdf_path), clean_temp=False)

        if result["status"] == "success":
            return UploadResponse(
                status="success",
                message=f"成功索引 PDF: {file.filename}",
                pages_processed=result["pages_processed"],
                total_docs=result["total_docs"]
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=f"索引失敗: {result.get('error', '未知錯誤')}"
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"上傳失敗: {str(e)}")


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    提交問題並獲取答案

    Args:
        request: 查詢請求，包含問題和 top_k

    Returns:
        答案和來源資訊
    """
    if not rag_pipeline:
        raise HTTPException(status_code=500, detail="RAG pipeline 未初始化")

    try:
        # 執行查詢
        result = rag_pipeline.query(request.question, top_k=request.top_k)

        if result["status"] == "success":
            return QueryResponse(
                status="success",
                answer=result["answer"],
                sources=result["sources"],
                question=result["question"]
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=f"查詢失敗: {result.get('error', '未知錯誤')}"
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查詢失敗: {str(e)}")


@app.get("/image/{filename}")
async def get_image(filename: str):
    """
    取得暫存圖像（用於顯示來源頁面）

    Args:
        filename: 圖像檔案名稱

    Returns:
        圖像檔案
    """
    image_path = config.TEMP_IMAGES_DIR / filename

    if not image_path.exists():
        raise HTTPException(status_code=404, detail="圖像不存在")

    return FileResponse(image_path)


@app.get("/stats")
async def get_stats():
    """
    取得系統統計資訊

    Returns:
        系統統計資料
    """
    if not rag_pipeline:
        raise HTTPException(status_code=500, detail="RAG pipeline 未初始化")

    return rag_pipeline.get_stats()


@app.delete("/reset")
async def reset_database():
    """
    重置資料庫（清除所有資料）

    Returns:
        重置結果
    """
    if not rag_pipeline:
        raise HTTPException(status_code=500, detail="RAG pipeline 未初始化")

    try:
        rag_pipeline.reset_database()
        return {"status": "success", "message": "資料庫已重置"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"重置失敗: {str(e)}")


@app.get("/health")
async def health_check():
    """健康檢查端點"""
    return {
        "status": "healthy",
        "rag_initialized": rag_pipeline is not None
    }


if __name__ == "__main__":
    import uvicorn

    print("\n啟動 FastAPI 應用...")
    print("訪問 http://localhost:8000 使用 Web 介面\n")

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
