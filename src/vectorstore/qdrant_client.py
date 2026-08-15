from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from src.config import settings

class VectorStoreManager:
    """Менеджер работы с векторной базой данных Qdrant."""

    def __init__(self):
        # Инициализируем Qdrant в локальной памяти/файле (не требует отдельного сервера)
        self.client = QdrantClient(location=":memory:")

    def init_collection(self, collection_name: str = settings.COLLECTION_NAME, vector_size: int = 512):
        """Создает коллекцию в Qdrant, если она еще не существует."""
        collections = [col.name for col in self.client.get_collections().collections]
        
        if collection_name not in collections:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )

    def upsert_vectors(self, collection_name: str, points: list):
        """Сохраняет эмбеддинги и метаданные в Qdrant."""
        self.client.upsert(
            collection_name=collection_name,
            points=points
        )
