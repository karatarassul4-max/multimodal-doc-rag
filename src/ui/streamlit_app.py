import sys
from pathlib import Path

# Добавляем корень проекта в sys.path для корректных импортов на Streamlit Cloud
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

import tempfile
import uuid
import streamlit as st
from qdrant_client.models import PointStruct

from src.embeddings.multimodal_encoder import MultimodalEncoder
from src.parser.pdf_parser import DocumentParser
from src.vectorstore.qdrant_client import VectorStoreManager

st.set_page_config(page_title="Multimodal Document RAG", layout="wide")

st.title("📄 Multimodal Document AI & RAG Search")
st.write("Загружайте PDF-документы и ищите информацию по тексту и изображениям.")

# Кешируем загрузку тяжелых моделей в Streamlit
@st.cache_resource
def load_components():
    encoder = MultimodalEncoder()
    # Используем локальное файловое хранилище для Qdrant в бесплатном Streamlit Cloud
    vector_store = VectorStoreManager()
    vector_store.init_collection(vector_size=512)
    return encoder, vector_store

encoder, vector_store = load_components()

st.sidebar.header("1. Загрузка документа")
uploaded_file = st.sidebar.file_uploader("Выберите PDF файл", type=["pdf"])

if uploaded_file is not None:
    if st.sidebar.button("Индексировать PDF"):
        with st.spinner("Парсинг и векторизация документа..."):
            # Сохраняем временный файл
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            try:
                parser = DocumentParser(tmp_path)
                pages = parser.extract_text_by_page()

                points = []
                for page in pages:
                    if not page["text"]:
                        continue

                    # Генерируем эмбеддинг
                    vector = encoder.encode_text(page["text"])
                    point_id = str(uuid.uuid4())

                    points.append(
                        PointStruct(
                            id=point_id,
                            vector=vector,
                            payload={
                                "filename": uploaded_file.name,
                                "page": page["page"],
                                "text": page["text"][:500]
                            }
                        )
                    )

                if points:
                    vector_store.upsert_vectors(
                        collection_name="multimodal_docs", 
                        points=points
                    )
                    st.sidebar.success(f"Успешно! Заиндексировано страниц: {len(points)}")
                else:
                    st.sidebar.warning("В файле не найдено текста для индексации.")
            except Exception as e:
                st.sidebar.error(f"Ошибка при обработке: {str(e)}")

st.header("2. Мультимодальный поиск")
query = st.text_input("Введите поисковый запрос (например: 'график продаж' или 'архитектура'):")

if st.button("Найти в документах") and query:
    with st.spinner("Поиск по векторной базе..."):
        try:
            query_vector = encoder.encode_text(query)

            # Используем актуальный метод query_points вместо устаревшего search
            search_result = vector_store.client.query_points(
                collection_name="multimodal_docs",
                query=query_vector,
                limit=3
            )

            st.subheader(f"Результаты по запросу: '{query}'")

            # Извлекаем точки из ответа query_points
            results = search_result.points

            if results:
                for idx, hit in enumerate(results):
                    payload = hit.payload or {}
                    filename = payload.get("filename", "Unknown")
                    page = payload.get("page", "-")
                    text = payload.get("text", "")

                    with st.expander(
                        f"Результат #{idx+1} | Файл: {filename} (Стр. {page}) | Score: {hit.score:.4f}"
                    ):
                        st.write("**Извлеченный фрагмент:**")
                        st.write(text)
            else:
                st.info("Ничего не найдено.")
        except Exception as e:
            st.error(f"Ошибка при выполнении поиска: {str(e)}")
