"""
摘要生成模組
使用 Gemma 3 Vision API 為 PDF 頁面圖像生成結構化摘要
"""
import base64
import time
from typing import List
from openai import OpenAI
import config


class SummaryGenerator:
    """摘要生成器類別"""

    def __init__(self):
        """初始化 OpenAI client"""
        self.client = OpenAI(
            api_key=config.LLM_API_KEY,
            base_url=config.LLM_API_BASE
        )
        self.model = config.LLM_MODEL_NAME
        self.max_retries = config.MAX_RETRIES

    def _encode_image(self, image_path: str) -> str:
        """
        將圖像編碼為 base64

        Args:
            image_path: 圖像檔案路徑

        Returns:
            base64 編碼的圖像字串
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def generate_summary(self, image_path: str) -> str:
        """
        為單個圖像生成摘要

        Args:
            image_path: 圖像檔案路徑

        Returns:
            摘要文字
        """
        # 將圖像編碼為 base64
        image_base64 = self._encode_image(image_path)

        # 重試機制
        for attempt in range(self.max_retries):
            try:
                print(f"正在生成摘要: {image_path} (嘗試 {attempt + 1}/{self.max_retries})")

                # 呼叫 Vision API
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": config.SUMMARY_PROMPT
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{image_base64}"
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=2000,
                    temperature=0.1  # 較低的溫度以獲得更穩定的輸出
                )

                summary = response.choices[0].message.content
                print(f"  摘要生成成功 (長度: {len(summary)} 字元)")
                return summary

            except Exception as e:
                print(f"  錯誤: {str(e)}")
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt  # 指數退避
                    print(f"  等待 {wait_time} 秒後重試...")
                    time.sleep(wait_time)
                else:
                    print(f"  達到最大重試次數，放棄")
                    raise Exception(f"無法為 {image_path} 生成摘要: {str(e)}")

    def batch_generate_summaries(self, image_paths: List[str]) -> List[str]:
        """
        批次生成摘要

        Args:
            image_paths: 圖像檔案路徑清單

        Returns:
            摘要文字清單
        """
        summaries = []
        total = len(image_paths)

        print(f"\n開始批次生成摘要，共 {total} 個頁面")

        for i, image_path in enumerate(image_paths, start=1):
            print(f"\n[{i}/{total}] 處理頁面...")
            summary = self.generate_summary(image_path)
            summaries.append(summary)

        print(f"\n批次處理完成！共生成 {len(summaries)} 個摘要")
        return summaries

    def generate_answer(self, query: str, image_paths: List[str]) -> str:
        """
        根據圖像回答問題

        Args:
            query: 使用者問題
            image_paths: 相關圖像路徑清單

        Returns:
            答案文字
        """
        # 準備問答 prompt
        prompt = config.QUERY_PROMPT_TEMPLATE.format(query=query)

        # 準備圖像內容（支援多張圖像）
        content = [{"type": "text", "text": prompt}]

        for image_path in image_paths:
            image_base64 = self._encode_image(image_path)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image_base64}"
                }
            })

        # 重試機制
        for attempt in range(self.max_retries):
            try:
                print(f"正在生成答案 (嘗試 {attempt + 1}/{self.max_retries})")

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": content
                        }
                    ],
                    max_tokens=1500,
                    temperature=0.3
                )

                answer = response.choices[0].message.content
                print(f"答案生成成功 (長度: {len(answer)} 字元)")
                return answer

            except Exception as e:
                print(f"錯誤: {str(e)}")
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"等待 {wait_time} 秒後重試...")
                    time.sleep(wait_time)
                else:
                    raise Exception(f"無法生成答案: {str(e)}")


if __name__ == "__main__":
    # 測試代碼
    import sys
    import os

    generator = SummaryGenerator()

    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        if os.path.exists(image_path):
            print(f"測試摘要生成: {image_path}\n")
            summary = generator.generate_summary(image_path)
            print(f"\n生成的摘要:\n{summary}")
        else:
            print(f"錯誤: 找不到圖像檔案 {image_path}")
    else:
        print("使用方式: python summary_generator.py <image_path>")
