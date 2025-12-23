"""
向量化模組
使用 BGE-M3 模型將文本轉換為向量嵌入
"""
import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer
import config


class BGEEmbedder:
    """BGE-M3 向量嵌入器"""

    def __init__(self, model_name: str = None):
        """
        初始化 BGE-M3 模型

        Args:
            model_name: 模型名稱，預設使用 config.BGE_MODEL_NAME
        """
        if model_name is None:
            model_name = config.BGE_MODEL_NAME

        print(f"正在載入 BGE-M3 模型: {model_name}")
        print("首次載入會自動下載模型（約 2GB），請稍候...")

        self.model = SentenceTransformer(model_name)
        self.dimension = config.VECTOR_DIM

        print(f"模型載入完成！向量維度: {self.dimension}")

    def encode(self, texts: List[str], batch_size: int = 32, show_progress: bool = True) -> np.ndarray:
        """
        批次編碼文本為向量

        Args:
            texts: 文本清單
            batch_size: 批次大小
            show_progress: 是否顯示進度

        Returns:
            向量陣列 (shape: [len(texts), dimension])
        """
        if not texts:
            return np.array([])

        print(f"正在編碼 {len(texts)} 個文本...")

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True  # BGE-M3 建議使用正規化
        )

        print(f"編碼完成！向量形狀: {embeddings.shape}")
        return embeddings

    def encode_single(self, text: str) -> np.ndarray:
        """
        編碼單個文本為向量

        Args:
            text: 文本字串

        Returns:
            向量陣列 (shape: [dimension])
        """
        embedding = self.model.encode(
            text,
            normalize_embeddings=True
        )
        return embedding

    def encode_queries(self, queries: List[str]) -> np.ndarray:
        """
        編碼查詢文本（專門用於檢索查詢）

        Args:
            queries: 查詢文本清單

        Returns:
            查詢向量陣列
        """
        # BGE-M3 對查詢和文檔使用相同的編碼方式
        return self.encode(queries, show_progress=False)

    def get_similarity(self, query_vector: np.ndarray, doc_vectors: np.ndarray) -> np.ndarray:
        """
        計算查詢向量與文檔向量的相似度

        Args:
            query_vector: 查詢向量 (shape: [dimension])
            doc_vectors: 文檔向量陣列 (shape: [n, dimension])

        Returns:
            相似度分數陣列 (shape: [n])
        """
        # 使用內積計算相似度（因為向量已經正規化）
        if doc_vectors.ndim == 1:
            doc_vectors = doc_vectors.reshape(1, -1)

        similarities = np.dot(doc_vectors, query_vector)
        return similarities


def test_embedder():
    """測試 embedder 功能"""
    print("=== 測試 BGE Embedder ===\n")

    # 初始化 embedder
    embedder = BGEEmbedder()

    # 測試文本（繁簡中文混合）
    test_texts = [
        "這是一個繁體中文的測試文本。",
        "这是一个简体中文的测试文本。",
        "This is an English test text.",
        "機器學習是人工智慧的一個重要分支。"
    ]

    print(f"\n測試文本:")
    for i, text in enumerate(test_texts, 1):
        print(f"  {i}. {text}")

    # 批次編碼
    print(f"\n批次編碼測試:")
    embeddings = embedder.encode(test_texts)
    print(f"結果形狀: {embeddings.shape}")
    print(f"第一個向量的前 10 維: {embeddings[0][:10]}")

    # 單個編碼
    print(f"\n單個編碼測試:")
    single_embedding = embedder.encode_single(test_texts[0])
    print(f"結果形狀: {single_embedding.shape}")
    print(f"前 10 維: {single_embedding[:10]}")

    # 相似度測試
    print(f"\n相似度測試:")
    query = "人工智慧和深度學習"
    query_vector = embedder.encode_single(query)

    similarities = embedder.get_similarity(query_vector, embeddings)
    print(f"查詢: {query}")
    print(f"與各文本的相似度:")
    for i, (text, sim) in enumerate(zip(test_texts, similarities), 1):
        print(f"  {i}. {text[:30]}... -> {sim:.4f}")

    print("\n測試完成！")


if __name__ == "__main__":
    test_embedder()
