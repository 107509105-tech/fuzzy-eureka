"""
PDF 文件處理模組
負責將 PDF 轉換為圖像並管理暫存圖像
"""
import os
import shutil
from pathlib import Path
from typing import List
from pdf2image import convert_from_path
import config


def pdf_to_images(pdf_path: str, output_dir: str = None) -> List[str]:
    """
    將 PDF 轉換為圖像（每頁一張）

    Args:
        pdf_path: PDF 檔案路徑
        output_dir: 輸出目錄，預設使用 config.TEMP_IMAGES_DIR

    Returns:
        圖像檔案路徑清單
    """
    if output_dir is None:
        output_dir = str(config.TEMP_IMAGES_DIR)

    # 確保輸出目錄存在
    os.makedirs(output_dir, exist_ok=True)

    # 取得 PDF 檔名（不含副檔名）
    pdf_name = Path(pdf_path).stem

    print(f"正在轉換 PDF: {pdf_path}")

    # 轉換 PDF 為圖像
    images = convert_from_path(
        pdf_path,
        dpi=config.PDF_DPI,
        fmt='png'
    )

    # 儲存圖像並記錄路徑
    image_paths = []
    for i, image in enumerate(images, start=1):
        image_filename = f"{pdf_name}_page_{i}.png"
        image_path = os.path.join(output_dir, image_filename)
        image.save(image_path, 'PNG')
        image_paths.append(image_path)
        print(f"  已處理: 第 {i}/{len(images)} 頁")

    print(f"完成！共轉換 {len(image_paths)} 頁")
    return image_paths


def clean_temp_images(output_dir: str = None):
    """
    清理暫存圖像目錄

    Args:
        output_dir: 要清理的目錄，預設使用 config.TEMP_IMAGES_DIR
    """
    if output_dir is None:
        output_dir = str(config.TEMP_IMAGES_DIR)

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        print(f"已清理暫存圖像目錄: {output_dir}")


def get_page_info(image_path: str) -> dict:
    """
    從圖像檔案路徑解析頁面資訊

    Args:
        image_path: 圖像檔案路徑

    Returns:
        包含 doc_name 和 page_num 的字典
    """
    filename = Path(image_path).stem
    # 檔名格式: {pdf_name}_page_{page_num}
    parts = filename.rsplit('_page_', 1)

    if len(parts) == 2:
        doc_name = parts[0]
        page_num = int(parts[1])
    else:
        doc_name = filename
        page_num = 1

    return {
        'doc_name': doc_name,
        'page_num': page_num
    }


if __name__ == "__main__":
    # 測試代碼
    import sys

    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        if os.path.exists(pdf_path):
            image_paths = pdf_to_images(pdf_path)
            print(f"\n生成的圖像:")
            for path in image_paths:
                info = get_page_info(path)
                print(f"  - {path}")
                print(f"    文件: {info['doc_name']}, 頁數: {info['page_num']}")
        else:
            print(f"錯誤: 找不到檔案 {pdf_path}")
    else:
        print("使用方式: python document_processor.py <pdf_path>")
