from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
from urllib.parse import quote
from typing import Optional

# Milvus 相關
from pymilvus import Collection, connections
from km import smart_step_query as km_query, get_embedding, _clean_user_input, process_image_info
from config import (
    MILVUS_HOST,
    MILVUS_PORT,
    MILVUS_DB_NAME,
    MILVUS_USER,
    MILVUS_PASSWORD,
)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# ====== 圖片內網 URL 基礎位置 ======
BASE_IMAGE_URL = "http://10.12.100.164:8001/static/images/"

collection: Collection  # 單一 collection 全域變數

@app.on_event("startup")
def startup() -> None:
    global collection
    connections.connect(
        alias="default",
        host=MILVUS_HOST,
        port=MILVUS_PORT,
        user=MILVUS_USER,
        password=MILVUS_PASSWORD,
        db_name=MILVUS_DB_NAME,
    )
    collection = Collection(name="drilling")

@app.get("/", response_class=HTMLResponse)
def serve_frontend() -> str:
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

class QueryRequest(BaseModel):
    query: str
    title: Optional[str] = None

@app.post("/query")
def query_law(data: QueryRequest) -> JSONResponse:
    try:
        raw_input = data.query
        if not raw_input:
            return JSONResponse({"result": "請輸入查詢內容"}, status_code=400)

        # 取得前端選的 title（可能是 None）
        title_filter = data.title

        # 呼叫新版 smart_step_query，將 title_filter 傳入
        # km_query 現在的簽名為:
        # smart_step_query(user_input, coll, embed_fn, top_k=20, title_filter=None)
        result = km_query(
            raw_input,
            collection,
            get_embedding,
            top_k=20,
            title_filter=title_filter,   # ← 關鍵：把使用者選的 title 傳下去
        )

        if not result:
            return JSONResponse({"result": []})

        # 已在前端處理過圖片 URL，這裡只需要回傳即可
        results_with_images = [process_image_info(hit, BASE_IMAGE_URL) for hit in result]
        return JSONResponse({"result": results_with_images})

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
