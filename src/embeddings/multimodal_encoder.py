from PIL import Image
from sentence_transformers import SentenceTransformer

class MultimodalEncoder:
    """Класс для перевода текста и изображений в единое векторное пространство (CLIP)."""

    def __init__(self, model_name: str = "clip-ViT-B-32"):
        # Используем предразмеченную мультимодальную модель CLIP
        self.model = SentenceTransformer(model_name)

    def encode_text(self, text: str) -> list[float]:
        """Преобразует текст в вектор."""
        embedding = self.model.encode(text)
        return embedding.tolist()

    def encode_image(self, image: Image.Image) -> list[float]:
        """Преобразует изображение (PIL Image) в вектор."""
        embedding = self.model.encode(image)
        return embedding.tolist()
