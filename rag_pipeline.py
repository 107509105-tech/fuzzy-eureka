"""
RAG 檢索問答模組
整合 PDF 處理、摘要生成、向量化和檢索的完整流程
"""
import os
from typing import Dict, List
from document_processor import pdf_to_images, get_page_info, clean_temp_images
from summary_generator import SummaryGenerator
from embedder import BGEEmbedder
from vector_store import MilvusStore
import config


class RAGPipeline:
    """RAG 系統主要流程管理器"""

    def __init__(self):
        """初始化 RAG pipeline 的所有組件"""
        print("=== 初始化 RAG Pipeline ===\n")

        # 初始化各組件
        print("1. 初始化向量嵌入器...")
        self.embedder = BGEEmbedder()

        print("\n2. 初始化摘要生成器...")
        self.summary_generator = SummaryGenerator()

        print("\n3. 初始化向量資料庫...")
        self.vector_store = MilvusStore()
        self.vector_store.create_collection(drop_existing=False)

        print("\n=== RAG Pipeline 初始化完成 ===\n")

    def index_pdf(self, pdf_path: str, clean_temp: bool = False) -> Dict:
        """
        索引單個 PDF 文件

        完整流程：
        1. PDF 轉圖像
        2. 生成摘要
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
            # Step 1: PDF 轉圖像
            print("步驟 1/4: 轉換 PDF 為圖像")
            print("-" * 40)
            image_paths = pdf_to_images(pdf_path)

            if not image_paths:
                raise ValueError("PDF 轉換失敗，未生成任何圖像")

            # Step 2: 生成摘要
            print(f"\n步驟 2/4: 生成頁面摘要")
            print("-" * 40)
            summaries = self.summary_generator.batch_generate_summaries(image_paths)

            # Step 3: 向量化摘要
            print(f"\n步驟 3/4: 向量化摘要")
            print("-" * 40)
            embeddings = self.embedder.encode(summaries)

            # Step 4: 存入 Milvus
            print(f"\n步驟 4/4: 存入向量資料庫")
            print("-" * 40)

            # 準備資料
            data = []
            for i, (image_path, summary, embedding) in enumerate(
                zip(image_paths, summaries, embeddings)
            ):
                page_info = get_page_info(image_path)
                data.append({
                    "embedding": embedding,
                    "text_summary": summary,
                    "page_num": page_info["page_num"],
                    "doc_name": page_info["doc_name"],
                    "image_path": image_path
                })

            # 插入資料
            ids = self.vector_store.insert(data)

            # 清理暫存圖像（可選）
            if clean_temp:
                print("\n清理暫存圖像...")
                clean_temp_images()

            # 返回結果
            result = {
                "status": "success",
                "pdf_path": pdf_path,
                "pages_processed": len(image_paths),
                "ids": ids,
                "total_docs": self.vector_store.count()
            }

            print(f"\n{'='*60}")
            print(f"索引完成！")
            print(f"  處理頁數: {result['pages_processed']}")
            print(f"  資料庫總文檔數: {result['total_docs']}")
            print(f"{'='*60}\n")

            return result

        except Exception as e:
            print(f"\n錯誤: 索引 PDF 失敗 - {str(e)}")
            return {
                "status": "error",
                "pdf_path": pdf_path,
                "error": str(e)
            }

    def query(self, question: str, top_k: int = None) -> Dict:
        """
        查詢問答

        完整流程：
        1. 問題向量化
        2. Milvus 相似度搜尋
        3. 組合檢索結果
        4. 呼叫 LLM 生成最終答案

        Args:
            question: 使用者問題
            top_k: 檢索文檔數量，預設使用 config.TOP_K

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
            print(f"問題向量形狀: {query_vector.shape}")

            # Step 2: 搜尋相關文檔
            print(f"\n步驟 2/3: 搜尋相關文檔 (top {top_k})")
            print("-" * 40)
            search_results = self.vector_store.search(query_vector, top_k=top_k)

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
                print(f"  {i}. {result['doc_name']} - 第 {result['page_num']} 頁 "
                      f"(相似度: {result['distance']:.4f})")

            # Step 3: 生成答案
            print(f"\n步驟 3/3: 生成答案")
            print("-" * 40)

            # 取得相關頁面的圖像路徑
            image_paths = [result["image_path"] for result in search_results]

            # 呼叫 LLM 生成答案
            answer = self.summary_generator.generate_answer(question, image_paths)

            # 準備來源資訊
            sources = []
            for result in search_results:
                sources.append({
                    "doc_name": result["doc_name"],
                    "page_num": result["page_num"],
                    "image_path": result["image_path"],
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

    def get_stats(self) -> Dict:
        """
        取得系統統計資訊

        Returns:
            統計資訊字典
        """
        return {
            "total_documents": self.vector_store.count(),
            "collection_name": self.vector_store.collection_name,
            "vector_dim": config.VECTOR_DIM,
            "embedding_model": config.BGE_MODEL_NAME
        }

    def reset_database(self):
        """重置資料庫（清除所有資料）"""
        print("警告: 即將清除所有資料！")
        self.vector_store.delete_collection()
        self.vector_store.create_collection(drop_existing=False)
        print("資料庫已重置")


def main():
    """主程式：示範完整流程"""
    import sys

    # 初始化 RAG pipeline
    rag = RAGPipeline()

    # 檢查命令行參數
    if len(sys.argv) < 2:
        print("使用方式:")
        print("  索引模式: python rag_pipeline.py index <pdf_path>")
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

    elif mode == "stats":
        # 統計模式
        stats = rag.get_stats()
        print("\n系統統計:")
        print(f"  總文檔數: {stats['total_documents']}")
        print(f"  Collection: {stats['collection_name']}")
        print(f"  向量維度: {stats['vector_dim']}")
        print(f"  嵌入模型: {stats['embedding_model']}")

    else:
        print(f"錯誤: 未知模式 '{mode}'")


if __name__ == "__main__":
    main()
