"""
Milvus 向量存儲模組
使用 Milvus Lite 作為本地向量資料庫
"""
import numpy as np
from typing import List, Dict
from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility
)
import config


class MilvusStore:
    """Milvus 向量存儲管理器"""

    def __init__(self, uri: str = None, collection_name: str = None):
        """
        初始化 Milvus 連接

        Args:
            uri: Milvus 連接 URI，預設使用 config.MILVUS_URI
            collection_name: Collection 名稱，預設使用 config.COLLECTION_NAME
        """
        if uri is None:
            uri = config.MILVUS_URI
        if collection_name is None:
            collection_name = config.COLLECTION_NAME

        self.uri = uri
        self.collection_name = collection_name
        self.collection = None

        # 連接到 Milvus Lite
        print(f"正在連接 Milvus Lite: {uri}")
        connections.connect(uri=uri)
        print("Milvus 連接成功！")

    def create_collection(self, drop_existing: bool = False):
        """
        建立 collection 和索引

        Args:
            drop_existing: 是否刪除已存在的 collection
        """
        # 檢查 collection 是否存在
        if utility.has_collection(self.collection_name):
            if drop_existing:
                print(f"刪除現有 collection: {self.collection_name}")
                utility.drop_collection(self.collection_name)
            else:
                print(f"Collection {self.collection_name} 已存在，載入中...")
                self.collection = Collection(self.collection_name)
                return

        print(f"建立新 collection: {self.collection_name}")

        # 定義 schema
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=config.VECTOR_DIM),
            FieldSchema(name="text_summary", dtype=DataType.VARCHAR, max_length=4096),
            FieldSchema(name="page_num", dtype=DataType.INT32),
            FieldSchema(name="doc_name", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="image_path", dtype=DataType.VARCHAR, max_length=512)
        ]

        schema = CollectionSchema(
            fields=fields,
            description="PDF RAG Collection"
        )

        # 建立 collection
        self.collection = Collection(
            name=self.collection_name,
            schema=schema
        )

        # 建立 HNSW 索引
        print("建立 HNSW 索引...")
        index_params = {
            "index_type": "HNSW",
            "metric_type": "IP",  # Inner Product，適合已正規化的向量
            "params": {
                "M": 16,
                "efConstruction": 200
            }
        }

        self.collection.create_index(
            field_name="embedding",
            index_params=index_params
        )

        print("Collection 和索引建立完成！")

    def insert(self, data: List[Dict]) -> List[int]:
        """
        插入向量資料

        Args:
            data: 資料清單，每個元素包含:
                  - embedding: np.ndarray
                  - text_summary: str
                  - page_num: int
                  - doc_name: str
                  - image_path: str

        Returns:
            插入的 ID 清單
        """
        if not self.collection:
            raise ValueError("Collection 未初始化，請先呼叫 create_collection()")

        if not data:
            return []

        print(f"正在插入 {len(data)} 筆資料...")

        # 準備資料
        embeddings = [d["embedding"].tolist() if isinstance(d["embedding"], np.ndarray)
                     else d["embedding"] for d in data]
        text_summaries = [d["text_summary"] for d in data]
        page_nums = [d["page_num"] for d in data]
        doc_names = [d["doc_name"] for d in data]
        image_paths = [d["image_path"] for d in data]

        # 插入資料
        entities = [
            embeddings,
            text_summaries,
            page_nums,
            doc_names,
            image_paths
        ]

        insert_result = self.collection.insert(entities)

        # Flush 確保資料寫入
        self.collection.flush()

        print(f"插入完成！共 {len(insert_result.primary_keys)} 筆資料")
        return insert_result.primary_keys

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = None,
        output_fields: List[str] = None
    ) -> List[Dict]:
        """
        相似度搜尋

        Args:
            query_vector: 查詢向量
            top_k: 返回結果數量，預設使用 config.TOP_K
            output_fields: 要返回的欄位清單

        Returns:
            搜尋結果清單，每個元素包含 id, distance 和 entity 欄位
        """
        if not self.collection:
            raise ValueError("Collection 未初始化")

        if top_k is None:
            top_k = config.TOP_K

        if output_fields is None:
            output_fields = ["text_summary", "page_num", "doc_name", "image_path"]

        # 載入 collection 到記憶體
        self.collection.load()

        print(f"正在搜尋 top {top_k} 相似結果...")

        # 執行搜尋
        search_params = {
            "metric_type": "IP",
            "params": {"ef": 100}
        }

        # 確保 query_vector 是正確格式
        if isinstance(query_vector, np.ndarray):
            query_vector = query_vector.tolist()

        results = self.collection.search(
            data=[query_vector],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=output_fields
        )

        # 格式化結果
        formatted_results = []
        for hits in results:
            for hit in hits:
                result = {
                    "id": hit.id,
                    "distance": hit.distance,
                    "text_summary": hit.entity.get("text_summary"),
                    "page_num": hit.entity.get("page_num"),
                    "doc_name": hit.entity.get("doc_name"),
                    "image_path": hit.entity.get("image_path")
                }
                formatted_results.append(result)

        print(f"搜尋完成！找到 {len(formatted_results)} 個結果")
        return formatted_results

    def count(self) -> int:
        """
        取得 collection 中的資料數量

        Returns:
            資料筆數
        """
        if not self.collection:
            return 0

        self.collection.flush()
        return self.collection.num_entities

    def delete_collection(self):
        """刪除 collection"""
        if utility.has_collection(self.collection_name):
            print(f"刪除 collection: {self.collection_name}")
            utility.drop_collection(self.collection_name)
            self.collection = None
            print("刪除完成！")
        else:
            print(f"Collection {self.collection_name} 不存在")

    def close(self):
        """關閉連接"""
        connections.disconnect(alias="default")
        print("Milvus 連接已關閉")


def test_vector_store():
    """測試 vector store 功能"""
    print("=== 測試 Milvus Vector Store ===\n")

    # 初始化
    store = MilvusStore()

    # 建立 collection（刪除現有）
    store.create_collection(drop_existing=True)

    # 準備測試資料
    test_data = [
        {
            "embedding": np.random.rand(config.VECTOR_DIM).astype(np.float32),
            "text_summary": "這是第一頁的摘要內容",
            "page_num": 1,
            "doc_name": "測試文件",
            "image_path": "/path/to/page_1.png"
        },
        {
            "embedding": np.random.rand(config.VECTOR_DIM).astype(np.float32),
            "text_summary": "這是第二頁的摘要內容",
            "page_num": 2,
            "doc_name": "測試文件",
            "image_path": "/path/to/page_2.png"
        }
    ]

    # 插入資料
    ids = store.insert(test_data)
    print(f"\n插入的 IDs: {ids}")

    # 檢查數量
    count = store.count()
    print(f"\nCollection 中的資料數量: {count}")

    # 搜尋測試
    query_vector = np.random.rand(config.VECTOR_DIM).astype(np.float32)
    results = store.search(query_vector, top_k=2)

    print(f"\n搜尋結果:")
    for i, result in enumerate(results, 1):
        print(f"\n結果 {i}:")
        print(f"  ID: {result['id']}")
        print(f"  相似度: {result['distance']:.4f}")
        print(f"  文件: {result['doc_name']}")
        print(f"  頁數: {result['page_num']}")
        print(f"  摘要: {result['text_summary'][:50]}...")

    # 清理
    # store.delete_collection()
    store.close()

    print("\n測試完成！")


if __name__ == "__main__":
    test_vector_store()
