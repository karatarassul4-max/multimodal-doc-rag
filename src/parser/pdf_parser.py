from pathlib import Path
import fitz  # PyMuPDF
from PIL import Image
import io

class DocumentParser:
    """Модуль для извлечения текста и рендеринга страниц PDF в изображения."""
    
    def __init__(self, pdf_path: str):
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"Файл {pdf_path} не найден.")

    def extract_text_by_page(self) -> list[dict]:
        """Извлекает чистый текст с каждой страницы."""
        doc = fitz.open(self.pdf_path)
        pages_data = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            pages_data.append({
                "page": page_num + 1,
                "text": text.strip()
            })
            
        doc.close()
        return pages_data

    def page_to_image(self, page_num: int, dpi: int = 200) -> Image.Image:
        """Рендерит указанную страницу PDF в PIL Image для дальнейшей работы Vision-модели."""
        doc = fitz.open(self.pdf_path)
        page = doc[page_num]
        pix = page.get_pixmap(dpi=dpi)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        doc.close()
        return img
