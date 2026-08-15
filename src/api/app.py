import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from qdrant_client.models import PointStruct

from src.parser.pdf_parser import DocumentParser
from src.embeddings.multimodal_encoder import MultimodalEncoder
from src.vectorstore.qdrant_client import VectorStoreManager
from src.config import settings

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Multimodal Document RAG API (PDF Parsing, Embeddings, Search)",
    version="0.1.0"
)

# Инициализируем компоненты
encoder = MultimodalEncoder()
vector_store = VectorStoreManager()

@app.on_event("startup")
def startup_event():
    # Создаем коллекцию с размерностью векторов CLIP ViT-B-32 (512)
    vector_store.init_collection(vector_size=512)

class SearchQuery(BaseModel):
    query: str
    top_k: int = 3

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """Загружает PDF, разбивает на страницы и индексирует в Qdrant."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Файл должен быть формата PDF")

    # Временно сохраняем файл
    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, "wb") as buffer:
        buffer.write(await file.read())

    parser = DocumentParser(temp_path)
    pages = parser.extract_text_by_page()

    points = []
    for page in pages:
        if not page["text"]:
            continue
            
        # Генерируем эмбеддинг текста страницы
        vector = encoder.encode_text(page["text"])
        point_id = str(uuid.uuid4())
        
        points.append(
            PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "filename": file.filename,
                    "page": page["page"],
                    "text": page["text"][:500]  # Храним превью текста
                }
            )
        )

    if points:
        vector_store.upsert_vectors(collection_name=settings.COLLECTION_NAME, points=points)

    return {"status": "success", "indexed_pages": len(points), "filename": file.filename}

@app.post("/search")
async def search_documents(payload: SearchQuery):
    """Мультимодальный поиск по векторизованным документам."""
    query_vector = encoder.encode_text(payload.query)
    
    results = vector_store.client.search(
        collection_name=settings.COLLECTION_NAME,
        query_vector=query_vector,
        limit=payload.top_k
    )

    return {
        "query": payload.query,
        "results": [
            {
                "score": hit.score,
                "filename": hit.payload.get("filename"),
                "page": hit.payload.get("page"),
                "text": hit.payload.get("text")
            }
            for hit in results
        ]
    }
