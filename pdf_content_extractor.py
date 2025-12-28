"""
增強版 PDF 內容提取器
支援混合內容提取：文字、表格、圖片的智能識別和處理
"""
import fitz  # PyMuPDF
import pdfplumber
from pathlib import Path
from typing import Dict, List, Tuple, Optional


# 頁面類型常數
PAGE_TYPE_TEXT_HEAVY = "text_heavy"    # 純文字為主 (>80% 文字)
PAGE_TYPE_TABLE_HEAVY = "table_heavy"  # 表格為主
PAGE_TYPE_IMAGE_HEAVY = "image_heavy"  # 圖片/掃描件為主 (<20% 可選文字)
PAGE_TYPE_MIXED = "mixed"              # 混合內容


class PageContent:
    """頁面內容結構"""

    def __init__(
        self,
        page_num: int,
        page_type: str,
        text: str,
        tables: List[List[List[str]]],
        images_count: int,
        has_native_text: bool,
        text_coverage: float,
        image_path: Optional[str] = None
    ):
        """
        初始化頁面內容

        Args:
            page_num: 頁碼
            page_type: 頁面類型
            text: 提取的文字內容
            tables: 表格數據列表
            images_count: 圖片數量
            has_native_text: 是否有原生可選文字
            text_coverage: 文字覆蓋率
            image_path: 頁面圖片路徑
        """
        self.page_num = page_num
        self.page_type = page_type
        self.text = text
        self.tables = tables
        self.images_count = images_count
        self.has_native_text = has_native_text
        self.text_coverage = text_coverage
        self.image_path = image_path

    def to_structured_text(self) -> str:
        """
        轉換為結構化文字格式，適合向量化
        優先使用提取的內容，必要時標註需要 Vision LLM
        """
        sections = []

        # 頁面基本資訊
        sections.append(f"=== 第 {self.page_num} 頁 ===\n")

        # 文字內容
        if self.text and len(self.text.strip()) > 50:
            sections.append("【文字內容】")
            sections.append(self.text.strip())
            sections.append("")

        # 表格內容
        if self.tables:
            sections.append("【表格內容】")
            for i, table in enumerate(self.tables, 1):
                sections.append(f"\n表格 {i}:")
                sections.append(self._format_table_markdown(table))
            sections.append("")

        # 圖片標註
        if self.images_count > 0:
            sections.append(f"【圖片】此頁包含 {self.images_count} 個圖片/圖表")
            if self.page_type == PAGE_TYPE_IMAGE_HEAVY:
                sections.append("（此頁面以圖片為主，建議使用 Vision LLM 進行詳細分析）")
            sections.append("")

        # 頁面類型標註
        sections.append(f"【頁面類型】{self.page_type}")

        return "\n".join(sections)

    def _format_table_markdown(self, table: List[List[str]]) -> str:
        """將表格轉換為 Markdown 格式"""
        if not table or len(table) == 0:
            return ""

        # 計算每列的最大寬度
        col_widths = []
        num_cols = max(len(row) for row in table) if table else 0

        for col_idx in range(num_cols):
            max_width = 0
            for row in table:
                if col_idx < len(row) and row[col_idx]:
                    max_width = max(max_width, len(str(row[col_idx])))
            col_widths.append(max(max_width, 3))

        # 格式化表格
        lines = []
        for i, row in enumerate(table):
            # 補齊缺少的列
            padded_row = row + [""] * (num_cols - len(row))
            # 格式化每個單元格
            formatted_cells = [
                str(cell).ljust(col_widths[j])
                for j, cell in enumerate(padded_row)
            ]
            lines.append("| " + " | ".join(formatted_cells) + " |")

            # 添加表頭分隔線
            if i == 0:
                separator = "| " + " | ".join(["-" * w for w in col_widths]) + " |"
                lines.append(separator)

        return "\n".join(lines)

    def needs_vision_llm(self) -> bool:
        """判斷是否需要 Vision LLM 處理"""
        return (
            self.page_type == PAGE_TYPE_IMAGE_HEAVY or
            (self.page_type == PAGE_TYPE_MIXED and self.images_count > 2) or
            (not self.has_native_text and self.images_count > 0)
        )


class PDFContentExtractor:
    """增強版 PDF 內容提取器"""

    def __init__(
        self,
        text_threshold: float = 0.2,      # 文字覆蓋率閾值
        table_detection: bool = True,      # 是否啟用表格檢測
        save_images: bool = True,          # 是否保存頁面圖片
        image_zoom: float = 2.0            # 圖片解析度
    ):
        """
        初始化提取器

        Args:
            text_threshold: 文字覆蓋率閾值，低於此值視為圖片頁面
            table_detection: 是否啟用表格檢測
            save_images: 是否保存頁面圖片（用於 Vision LLM）
            image_zoom: 圖片解析度倍率
        """
        self.text_threshold = text_threshold
        self.table_detection = table_detection
        self.save_images = save_images
        self.image_zoom = image_zoom

    def analyze_page_type(
        self,
        text_length: int,
        tables_count: int,
        images_count: int,
        text_coverage: float
    ) -> str:
        """
        分析頁面類型

        Args:
            text_length: 文字長度
            tables_count: 表格數量
            images_count: 圖片數量
            text_coverage: 文字覆蓋率

        Returns:
            頁面類型字串
        """
        # 圖片為主：文字覆蓋率低且有圖片
        if text_coverage < self.text_threshold and images_count > 0:
            return PAGE_TYPE_IMAGE_HEAVY

        # 表格為主：有表格且文字不多
        if tables_count > 0 and text_length < 1000:
            return PAGE_TYPE_TABLE_HEAVY

        # 文字為主：文字覆蓋率高
        if text_coverage > 0.6 and images_count <= 1:
            return PAGE_TYPE_TEXT_HEAVY

        # 混合頁面
        return PAGE_TYPE_MIXED

    def extract_text_from_page(self, page: fitz.Page) -> Tuple[str, float]:
        """
        從頁面提取文字

        Args:
            page: PyMuPDF 頁面對象

        Returns:
            (文字內容, 文字覆蓋率)
        """
        # 提取文字
        text = page.get_text("text")

        # 計算文字覆蓋率（文字塊面積 / 頁面面積）
        text_blocks = page.get_text("blocks")  # 獲取文字塊
        page_rect = page.rect
        page_area = page_rect.width * page_rect.height

        text_area = 0
        for block in text_blocks:
            if block[6] == 0:  # 類型 0 表示文字塊
                x0, y0, x1, y1 = block[:4]
                text_area += (x1 - x0) * (y1 - y0)

        coverage = min(text_area / page_area, 1.0) if page_area > 0 else 0.0

        return text, coverage

    def extract_tables_from_page(
        self,
        pdf_path: str,
        page_num: int
    ) -> List[List[List[str]]]:
        """
        從頁面提取表格（使用 pdfplumber）

        Args:
            pdf_path: PDF 檔案路徑
            page_num: 頁碼（1-based）

        Returns:
            表格列表
        """
        if not self.table_detection:
            return []

        try:
            tables = []
            with pdfplumber.open(pdf_path) as pdf:
                # pdfplumber 使用 0-based 索引
                page = pdf.pages[page_num - 1]

                # 提取所有表格
                extracted_tables = page.extract_tables()

                if extracted_tables:
                    for table in extracted_tables:
                        # 清理表格數據（移除 None 值）
                        cleaned_table = []
                        for row in table:
                            cleaned_row = [
                                str(cell).strip() if cell is not None else ""
                                for cell in row
                            ]
                            cleaned_table.append(cleaned_row)

                        # 過濾空表格
                        if any(any(cell for cell in row) for row in cleaned_table):
                            tables.append(cleaned_table)

            return tables

        except Exception as e:
            print(f"  警告: 表格提取失敗 (第 {page_num} 頁): {str(e)}")
            return []

    def count_images_in_page(self, page: fitz.Page) -> int:
        """
        計算頁面中的圖片數量

        Args:
            page: PyMuPDF 頁面對象

        Returns:
            圖片數量
        """
        try:
            image_list = page.get_images(full=True)
            return len(image_list)
        except Exception:
            return 0

    def save_page_image(
        self,
        page: fitz.Page,
        output_path: str
    ) -> str:
        """
        保存頁面為圖片

        Args:
            page: PyMuPDF 頁面對象
            output_path: 輸出路徑

        Returns:
            圖片路徑
        """
        mat = fitz.Matrix(self.image_zoom, self.image_zoom)
        pix = page.get_pixmap(matrix=mat)
        pix.save(output_path)
        return output_path

    def extract_page_content(
        self,
        pdf_path: str,
        page_num: int,
        output_dir: Optional[Path] = None
    ) -> PageContent:
        """
        提取單個頁面的完整內容

        Args:
            pdf_path: PDF 檔案路徑
            page_num: 頁碼（1-based）
            output_dir: 圖片輸出目錄（可選）

        Returns:
            PageContent 對象
        """
        pdf_path = str(pdf_path)
        doc = fitz.open(pdf_path)

        try:
            # 獲取頁面（轉換為 0-based）
            page = doc.load_page(page_num - 1)

            # 1. 提取文字
            text, text_coverage = self.extract_text_from_page(page)

            # 2. 提取表格
            tables = self.extract_tables_from_page(pdf_path, page_num)

            # 3. 計算圖片數量
            images_count = self.count_images_in_page(page)

            # 4. 保存頁面圖片（如果需要）
            image_path = None
            if self.save_images and output_dir:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)

                doc_name = Path(pdf_path).stem.replace(" ", "_")
                image_filename = f"{doc_name}_page_{page_num}.png"
                image_path = str(output_dir / image_filename)
                self.save_page_image(page, image_path)

            # 5. 分析頁面類型
            page_type = self.analyze_page_type(
                text_length=len(text),
                tables_count=len(tables),
                images_count=images_count,
                text_coverage=text_coverage
            )

            # 6. 組合結果
            content = PageContent(
                page_num=page_num,
                page_type=page_type,
                text=text,
                tables=tables,
                images_count=images_count,
                has_native_text=len(text.strip()) > 50,
                text_coverage=text_coverage,
                image_path=image_path
            )

            return content

        finally:
            doc.close()

    def extract_pdf_content(
        self,
        pdf_path: str,
        output_dir: Optional[Path] = None,
        page_range: Optional[Tuple[int, int]] = None
    ) -> List[PageContent]:
        """
        提取整個 PDF 的內容

        Args:
            pdf_path: PDF 檔案路徑
            output_dir: 圖片輸出目錄
            page_range: 頁面範圍 (start, end)，1-based，包含端點

        Returns:
            PageContent 列表
        """
        pdf_path = str(pdf_path)
        doc = fitz.open(pdf_path)
        total_pages = doc.page_count
        doc.close()

        # 確定處理的頁面範圍
        if page_range is None:
            start, end = 1, total_pages
        else:
            start, end = page_range
            start = max(1, start)
            end = min(total_pages, end)

        print(f"\n開始提取 PDF 內容: {Path(pdf_path).name}")
        print(f"總頁數: {total_pages}, 處理範圍: {start}-{end}")
        print("-" * 60)

        contents = []
        for page_num in range(start, end + 1):
            print(f"\n[{page_num}/{end}] 處理第 {page_num} 頁...")

            content = self.extract_page_content(pdf_path, page_num, output_dir)

            # 輸出分析結果
            print(f"  類型: {content.page_type}")
            print(f"  文字: {len(content.text)} 字元 (覆蓋率: {content.text_coverage:.1%})")
            print(f"  表格: {len(content.tables)} 個")
            print(f"  圖片: {content.images_count} 個")

            if content.needs_vision_llm():
                print(f"  ⚠️  建議使用 Vision LLM 處理")

            contents.append(content)

        print(f"\n{'='*60}")
        print(f"提取完成！共處理 {len(contents)} 頁")

        # 統計
        type_counts = {}
        vision_needed = 0
        for content in contents:
            type_counts[content.page_type] = type_counts.get(content.page_type, 0) + 1
            if content.needs_vision_llm():
                vision_needed += 1

        print(f"\n頁面類型統計:")
        for page_type, count in type_counts.items():
            print(f"  {page_type}: {count} 頁")
        print(f"\n需要 Vision LLM: {vision_needed} 頁 ({vision_needed/len(contents):.1%})")
        print(f"{'='*60}\n")

        return contents


# 便捷函數
def extract_pdf_with_analysis(
    pdf_path: str,
    output_dir: str,
    **kwargs
) -> List[PageContent]:
    """
    便捷函數：提取 PDF 並返回分析結果

    Args:
        pdf_path: PDF 檔案路徑
        output_dir: 輸出目錄
        **kwargs: 傳遞給 PDFContentExtractor 的參數

    Returns:
        PageContent 列表
    """
    extractor = PDFContentExtractor(**kwargs)
    return extractor.extract_pdf_content(pdf_path, Path(output_dir))


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("使用方式: python pdf_content_extractor.py <pdf_path> [output_dir]")
        print("\n範例:")
        print("  python pdf_content_extractor.py manual.pdf ./output")
        sys.exit(1)

    pdf_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./static/pages"

    # 執行提取
    contents = extract_pdf_with_analysis(
        pdf_path=pdf_path,
        output_dir=output_dir,
        text_threshold=0.2,
        table_detection=True,
        save_images=True,
        image_zoom=2.0
    )

    # 顯示範例結果
    if contents:
        print("\n" + "="*60)
        print("範例：第一頁的結構化內容")
        print("="*60)
        print(contents[0].to_structured_text())
