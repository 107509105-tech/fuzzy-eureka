"""
RAG 檢索問答模組
整合 PDF 處理、摘要生成、向量化和檢索的完整流程
"""
import os
from pathlib import Path
from typing import Dict, List, Optional
# from document_processor import pdf_to_images, get_page_info, clean_temp_images
from pdf_image import pdf_to_images, get_page_info
from pdf_content_extractor import PDFContentExtractor, PAGE_TYPE_IMAGE_HEAVY, PAGE_TYPE_MIXED
from docx_content_extractor import DocxContentExtractor
from summary_generator import SummaryGenerator
from embedder import BGEEmbedder
from vector_store import MilvusStore
import config

STATIC_DIR = config.STATIC_DIR.resolve()


class RAGPipeline:
    """RAG 系統主要流程管理器"""

    @staticmethod
    def _to_relative_path(file_path: str) -> str:
        """Convert an absolute file path to a relative path from STATIC_DIR."""
        try:
            rel = Path(file_path).resolve().relative_to(STATIC_DIR)
            return rel.as_posix()
        except Exception:
            return file_path
    
    @staticmethod
    def _to_static_url(file_path: str) -> str | None:
        """Convert an on-disk static file path to a URL for the frontend.

        If `config.BASE_IMAGE_URL` is set, build an absolute URL based on it.
        Otherwise, default to the app-local `/static/` path.
        """
        try:
            # Normalize to a path relative to STATIC_DIR
            if not Path(file_path).is_absolute():
                rel_path = Path(file_path).as_posix()
            else:
                rel = Path(file_path).resolve().relative_to(STATIC_DIR)
                rel_path = rel.as_posix()

            base = getattr(config, "BASE_IMAGE_URL", "").rstrip("/")
            if base:
                # Avoid duplicating the 'pages' segment if BASE_IMAGE_URL already includes it
                if rel_path.startswith("pages/") and base.endswith("/pages"):
                    return f"{base}/{rel_path.split('pages/', 1)[1]}"
                else:
                    return f"{base}/{rel_path}"
            # Fallback to local static mount
            return f"/static/{rel_path}"
        except Exception:
            return None

    @staticmethod
    def _to_local_static_url(file_path: str) -> str | None:
        """Always return the app-local `/static/` URL for a static file."""
        try:
            if not Path(file_path).is_absolute():
                rel_path = Path(file_path).as_posix()
            else:
                rel = Path(file_path).resolve().relative_to(STATIC_DIR)
                rel_path = rel.as_posix()
            return f"/static/{rel_path}"
        except Exception:
            return None

    @staticmethod
    def _to_absolute_static_path(file_path: str) -> str:
        """Convert a relative static path (e.g. 'pages/…') to an absolute filesystem path.

        If `file_path` is already absolute, returns it unchanged.
        """
        p = Path(file_path)
        if p.is_absolute():
            return p.as_posix()
        return (STATIC_DIR / p).resolve().as_posix()

    def __init__(self):
        """初始化 RAG pipeline 的所有組件"""
        print("=== 初始化 RAG Pipeline ===\n")

        # 初始化各組件
        print("1. 初始化向量嵌入器...")
        self.embedder = BGEEmbedder()

        print("\n2. 初始化摘要生成器...")
        self.summary_generator = SummaryGenerator()

        print("\n3. 初始化 PDF 內容提取器...")
        self.pdf_extractor = PDFContentExtractor(
            text_threshold=0.2,
            table_detection=True,
            save_images=True,
            image_zoom=2.0
        )

        print("\n4. 初始化 DOCX 內容提取器...")
        self.docx_extractor = DocxContentExtractor()

        print("\n5. 初始化向量資料庫...")
        self.vector_store = MilvusStore()
        self.vector_store.create_collection(drop_existing=False)

        print("\n=== RAG Pipeline 初始化完成 ===\n")

    def index_pdf(self, pdf_path: str, clean_temp: bool = False) -> Dict:
        """
        索引單個 PDF 文件（使用智能混合提取）

        完整流程：
        1. 智能提取頁面內容（文字、表格、圖片）
        2. 根據頁面類型決定處理方式：
           - 文字/表格頁面 → 直接使用結構化文字
           - 圖片頁面 → 使用 Vision LLM
        3. 向量化
        4. 存入 Milvus

        Args:
            pdf_path: PDF 檔案路徑
            clean_temp: 完成後是否清理暫存圖像

        Returns:
            索引結果字典，包含處理的頁數和狀態
        """
        print(f"\n{'='*60}")
        print(f"開始索引 PDF: {pdf_path}")
        print(f"{'='*60}\n")

        try:
            # Step 1: 智能提取 PDF 內容
            print("步驟 1/4: 智能提取 PDF 內容（文字、表格、圖片）")
            print("-" * 40)

            output_dir = config.STATIC_DIR / "pages"
            page_contents = self.pdf_extractor.extract_pdf_content(
                pdf_path=pdf_path,
                output_dir=output_dir
            )

            if not page_contents:
                raise ValueError("PDF 提取失敗，未生成任何內容")

            # Step 2: 根據頁面類型生成摘要
            print(f"\n步驟 2/4: 生成頁面摘要（智能混合處理）")
            print("-" * 40)

            summaries = []
            vision_llm_count = 0

            for i, content in enumerate(page_contents, 1):
                print(f"\n[{i}/{len(page_contents)}] 處理第 {content.page_num} 頁...")

                if content.needs_vision_llm():
                    # 使用 Vision LLM 處理圖片為主的頁面
                    print(f"  → 使用 Vision LLM 處理（{content.page_type}）")
                    summary = self.summary_generator.generate_summary(content.image_path)
                    vision_llm_count += 1
                else:
                    # 直接使用結構化文字
                    print(f"  → 使用結構化文字（{content.page_type}）")
                    summary = content.to_structured_text()

                summaries.append(summary)
                print(f"  ✓ 摘要長度: {len(summary)} 字元")

            print(f"\n總結: 使用 Vision LLM: {vision_llm_count}/{len(page_contents)} 頁")

            # Step 3: 向量化摘要
            print(f"\n步驟 3/4: 向量化摘要")
            print("-" * 40)
            embeddings = self.embedder.encode(summaries)
            print(f"✓ 生成 {len(embeddings)} 個向量")

            # Step 4: 存入 Milvus
            print(f"\n步驟 4/4: 存入向量資料庫")
            print("-" * 40)

            # 準備資料
            data = []
            doc_name = Path(pdf_path).stem.replace(" ", "_")

            for content, summary, embedding in zip(page_contents, summaries, embeddings):
                # 轉換為相對路徑
                rel_image_path = self._to_relative_path(content.image_path) if content.image_path else ""

                data.append({
                    "embedding": embedding,
                    "text_summary": summary,
                    "page_num": content.page_num,
                    "doc_name": doc_name,
                    "doc_type": "pdf",
                    "image_path": rel_image_path
                })

            # 插入資料
            ids = self.vector_store.insert(data)
            print(f"✓ 成功插入 {len(ids)} 筆資料")

            # 返回結果
            result = {
                "status": "success",
                "pdf_path": pdf_path,
                "pages_processed": len(page_contents),
                "vision_llm_used": vision_llm_count,
                "ids": ids,
                "total_docs": self.vector_store.count()
            }

            print(f"\n{'='*60}")
            print(f"索引完成！")
            print(f"  處理頁數: {result['pages_processed']}")
            print(f"  Vision LLM 使用: {vision_llm_count} 頁")
            print(f"  資料庫總文檔數: {result['total_docs']}")
            print(f"{'='*60}\n")

            return result

        except Exception as e:
            print(f"\n錯誤: 索引 PDF 失敗 - {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "status": "error",
                "pdf_path": pdf_path,
                "error": str(e)
            }

    def index_docx(self, docx_path: str) -> Dict:
        """
        索引單個 DOCX 文件

        完整流程：
        1. 提取 DOCX 內容（文字、表格、圖片）
        2. 根據頁面類型決定處理方式：
           - 文字/表格頁面 → 直接使用結構化文字
           - 圖片頁面 → 使用 Vision LLM（如果有預覽圖）
        3. 向量化
        4. 存入 Milvus

        Args:
            docx_path: DOCX 檔案路徑

        Returns:
            索引結果字典，包含處理的頁數和狀態
        """
        print(f"\n{'='*60}")
        print(f"開始索引 DOCX: {docx_path}")
        print(f"{'='*60}\n")

        try:
            # Step 1: 提取 DOCX 內容
            print("步驟 1/4: 提取 DOCX 內容（文字、表格、圖片）")
            print("-" * 40)

            doc_name = Path(docx_path).stem.replace(" ", "_")
            page_contents = self.docx_extractor.extract(docx_path, doc_name)

            if not page_contents:
                raise ValueError("DOCX 提取失敗，未生成任何內容")

            # Step 2: 生成摘要
            print(f"\n步驟 2/4: 生成頁面摘要")
            print("-" * 40)

            summaries = []
            vision_llm_count = 0

            for i, content in enumerate(page_contents, 1):
                print(f"\n[{i}/{len(page_contents)}] 處理內容...")

                # 如果圖片很多且有預覽圖，使用 Vision LLM
                if content.images_count > 3 and content.image_path:
                    print(f"  → 使用 Vision LLM 處理（圖片較多: {content.images_count} 張）")
                    summary = self.summary_generator.generate_summary(content.image_path)
                    vision_llm_count += 1
                else:
                    # 使用文字內容
                    print(f"  → 使用結構化文字（{content.page_type}）")
                    summary = content.text

                summaries.append(summary)
                print(f"  ✓ 摘要長度: {len(summary)} 字元")

            print(f"\n總結: 使用 Vision LLM: {vision_llm_count}/{len(page_contents)} 頁")

            # Step 3: 向量化摘要
            print(f"\n步驟 3/4: 向量化摘要")
            print("-" * 40)
            embeddings = self.embedder.encode(summaries)
            print(f"✓ 生成 {len(embeddings)} 個向量")

            # Step 4: 存入 Milvus
            print(f"\n步驟 4/4: 存入向量資料庫")
            print("-" * 40)

            # 準備資料
            data = []
            for content, summary, embedding in zip(page_contents, summaries, embeddings):
                # 轉換為相對路徑
                rel_image_path = self._to_relative_path(content.image_path) if content.image_path else ""

                data.append({
                    "embedding": embedding,
                    "text_summary": summary,
                    "page_num": content.page_num,
                    "doc_name": doc_name,
                    "doc_type": "docx",
                    "image_path": rel_image_path
                })

            # 插入資料
            ids = self.vector_store.insert(data)
            print(f"✓ 成功插入 {len(ids)} 筆資料")

            # 返回結果
            result = {
                "status": "success",
                "docx_path": docx_path,
                "pages_processed": len(page_contents),
                "vision_llm_used": vision_llm_count,
                "ids": ids,
                "total_docs": self.vector_store.count()
            }

            print(f"\n{'='*60}")
            print(f"索引完成！")
            print(f"  處理內容數: {result['pages_processed']}")
            print(f"  Vision LLM 使用: {vision_llm_count} 頁")
            print(f"  資料庫總文檔數: {result['total_docs']}")
            print(f"{'='*60}\n")

            return result

        except Exception as e:
            print(f"\n錯誤: 索引 DOCX 失敗 - {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "status": "error",
                "docx_path": docx_path,
                "error": str(e)
            }

    def index_document(self, file_path: str) -> Dict:
        """
        自動檢測文件類型並索引（支援 PDF 和 DOCX）

        Args:
            file_path: 文件路徑

        Returns:
            索引結果字典
        """
        file_ext = Path(file_path).suffix.lower()

        if file_ext == ".pdf":
            return self.index_pdf(file_path)
        elif file_ext in [".docx", ".doc"]:
            return self.index_docx(file_path)
        else:
            return {
                "status": "error",
                "file_path": file_path,
                "error": f"不支援的文件類型: {file_ext}"
            }

    def query(self, question: str, top_k: int = None, doc_filter: Optional[str] = None) -> Dict:
        """
        查詢問答

        完整流程：
        1. 問題向量化
        2. Milvus 相似度搜尋（可選文件過濾）
        3. 組合檢索結果
        4. 呼叫 LLM 生成最終答案

        Args:
            question: 使用者問題
            top_k: 檢索文檔數量，預設使用 config.TOP_K
            doc_filter: 文件過濾條件，例如 'doc_name == "manual"' 或 'doc_type == "pdf"'

        Returns:
            答案字典，包含答案文字、來源頁面和相似度分數
        """
        if top_k is None:
            top_k = config.TOP_K

        print(f"\n{'='*60}")
        print(f"問題: {question}")
        print(f"{'='*60}\n")

        try:
            # Step 1: 問題向量化
            print("步驟 1/3: 向量化問題")
            print("-" * 40)
            query_vector = self.embedder.encode_single(question)
            # print(f"問題向量形狀: {query_vector.shape}")

            # Step 2: 搜尋相關文檔
            print(f"\n步驟 2/3: 搜尋相關文檔 (top {top_k})")
            if doc_filter:
                print(f"  過濾條件: {doc_filter}")
            print("-" * 40)
            search_results = self.vector_store.search(query_vector, top_k=top_k, filter_expr=doc_filter)

            if not search_results:
                return {
                    "status": "success",
                    "answer": "抱歉，我在資料庫中找不到相關資訊來回答您的問題。",
                    "sources": [],
                    "question": question
                }

            # 顯示檢索結果
            print(f"\n找到 {len(search_results)} 個相關文檔:")
            for i, result in enumerate(search_results, 1):
                print(f"  {i}. [{result.get('doc_type', 'pdf')}] {result['doc_name']} - 第 {result['page_num']} 頁 "
                      f"(相似度: {result['distance']:.4f})")

            # Step 3: 生成答案
            print(f"\n步驟 3/3: 生成答案")
            print("-" * 40)

            # 取得相關頁面的圖像路徑和摘要
            # Use absolute filesystem paths for LLM vision input
            image_paths = [self._to_absolute_static_path(result["image_path"]) for result in search_results]
            summaries = [result["text_summary"] for result in search_results]

            # 呼叫 LLM 生成答案（使用圖像和摘要）
            answer = self.summary_generator.generate_answer(question, image_paths, summaries)

            # 準備來源資訊
            sources = []
            for result in search_results:
                # image_path 已經是相對路徑
                rel_image_path = result["image_path"]
                sources.append({
                    "doc_name": result["doc_name"],
                    "doc_type": result.get("doc_type", "pdf"),
                    "page_num": result["page_num"],
                    # Provide local static URL as fallback for frontend
                    "image_path": self._to_local_static_url(rel_image_path),
                    "image_url": self._to_static_url(rel_image_path),
                    "similarity": float(result["distance"]),
                    "summary": result["text_summary"]
                })

            result = {
                "status": "success",
                "answer": answer,
                "sources": sources,
                "question": question
            }

            print(f"\n{'='*60}")
            print(f"答案生成完成！")
            print(f"{'='*60}\n")

            return result

        except Exception as e:
            print(f"\n錯誤: 查詢失敗 - {str(e)}")
            return {
                "status": "error",
                "question": question,
                "error": str(e)
            }

    # def get_stats(self) -> Dict:
    #     """
    #     取得系統統計資訊

    #     Returns:
    #         統計資訊字典
    #     """
    #     return {
    #         "total_documents": self.vector_store.count(),
    #         "collection_name": self.vector_store.collection_name,
    #         "vector_dim": config.VECTOR_DIM,
    #         "embedding_model": config.BGE_MODEL_NAME
    #     }

    # def reset_database(self):
    #     """重置資料庫（清除所有資料）"""
    #     print("警告: 即將清除所有資料！")
    #     self.vector_store.delete_collection()
    #     self.vector_store.create_collection(drop_existing=False)
    #     print("資料庫已重置")


def main():
    """主程式：示範完整流程"""
    import sys

    # 初始化 RAG pipeline
    rag = RAGPipeline()

    # 檢查命令行參數
    if len(sys.argv) < 2:
        print("使用方式:")
        print("  索引模式: python rag_pipeline.py index <pdf_path>")
        # python rag_pipeline.py index "I:\KM\manual.pdf"
        print("  查詢模式: python rag_pipeline.py query <question>")
        print("  統計模式: python rag_pipeline.py stats")
        return

    mode = sys.argv[1]

    if mode == "index":
        # 索引模式
        if len(sys.argv) < 3:
            print("錯誤: 請提供 PDF 路徑")
            return

        pdf_path = sys.argv[2]
        if not os.path.exists(pdf_path):
            print(f"錯誤: 找不到檔案 {pdf_path}")
            return

        result = rag.index_pdf(pdf_path)

        if result["status"] == "success":
            print(f"\n索引成功！處理了 {result['pages_processed']} 頁")
        else:
            print(f"\n索引失敗: {result['error']}")

    elif mode == "query":
        # 查詢模式
        if len(sys.argv) < 3:
            print("錯誤: 請提供問題")
            return

        question = " ".join(sys.argv[2:])
        result = rag.query(question)

        if result["status"] == "success":
            print(f"\n問題: {result['question']}")
            print(f"\n答案:\n{result['answer']}")
            print(f"\n來源:")
            for i, source in enumerate(result["sources"], 1):
                print(f"  {i}. {source['doc_name']} - 第 {source['page_num']} 頁 "
                      f"(相似度: {source['similarity']:.4f})")
        else:
            print(f"\n查詢失敗: {result['error']}")

    # elif mode == "stats":
    #     # 統計模式
    #     stats = rag.get_stats()
    #     print("\n系統統計:")
    #     print(f"  總文檔數: {stats['total_documents']}")
    #     print(f"  Collection: {stats['collection_name']}")
    #     print(f"  向量維度: {stats['vector_dim']}")
    #     print(f"  嵌入模型: {stats['embedding_model']}")

    # else:
    #     print(f"錯誤: 未知模式 '{mode}'")


if __name__ == "__main__":
    main()