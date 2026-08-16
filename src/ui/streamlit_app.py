import sys
from pathlib import Path

# Добавляем корень проекта в sys.path для корректных импортов
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

st.set_page_config(page_title="Multimodal Document AI & RAG", layout="wide")

st.title("📄 Multimodal Document AI & Explainer")
st.write("Загружайте PDF-документы (отчеты, учебники, статьи) и получайте подробное объяснение и разбор материала от AI.")

# Кешируем загрузку моделей
@st.cache_resource
def load_components():
    encoder = MultimodalEncoder()
    vector_store = VectorStoreManager()
    vector_store.init_collection(collection_name="multimodal_docs", vector_size=512)
    return encoder, vector_store

encoder, vector_store = load_components()

# --- Настройки API ключа в сайдбаре ---
st.sidebar.header("⚙️ Настройки LLM")
groq_api_key = st.sidebar.text_input("Groq API Key (опционально):", type="password", help="Бесплатный ключ от console.groq.com для включения LLM-генерации")

# --- Секция 1: Загрузка и индексация документов ---
st.sidebar.header("1. Загрузка документа")
uploaded_file = st.sidebar.file_uploader("Выберите PDF файл", type=["pdf"])

if uploaded_file is not None:
    if st.sidebar.button("Индексировать PDF"):
        with st.spinner("Парсинг и векторизация документа..."):
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

                    vector = encoder.encode_text(page["text"])
                    point_id = str(uuid.uuid4())

                    points.append(
                        PointStruct(
                            id=point_id,
                            vector=vector,
                            payload={
                                "filename": uploaded_file.name,
                                "page": page["page"],
                                "text": page["text"]
                            }
                        )
                    )

                if points:
                    vector_store.upsert_vectors(
                        collection_name="multimodal_docs", 
                        points=points
                    )
                    st.sidebar.success(f"Заиндексировано страниц: {len(points)}")
                else:
                    st.sidebar.warning("В файле не найдено текста.")
            except Exception as e:
                st.sidebar.error(f"Ошибка при обработке: {str(e)}")

# --- Секция 2: Подробный AI-разбор ---
st.header("2. Задать вопрос или запросить разбор темы")
query = st.text_input("О чём вы хотите узнать подробно из документа?", placeholder="Например: Объясни подробно архитектуру модели или распиши ключевые финансовые показатели")

top_k = st.slider("Количество релевантных страниц для анализа:", min_value=1, max_value=5, value=3)

if st.button("🚀 Объяснить подробно (AI Analysis)") and query:
    with st.spinner("Поиск информации и формирование подробного разбора..."):
        try:
            query_vector = encoder.encode_text(query)

            search_result = vector_store.client.query_points(
                collection_name="multimodal_docs",
                query=query_vector,
                limit=top_k
            )

            results = search_result.points

            if results:
                # Формируем единый контекст из найденных страниц
                context_blocks = []
                for idx, hit in enumerate(results):
                    payload = hit.payload or {}
                    page_num = payload.get("page", "?")
                    text = payload.get("text", "")
                    context_blocks.append(f"--- Страница {page_num} ---\n{text}")

                full_context = "\n\n".join(context_blocks)

                # Системный промпт для детального объяснения
                prompt = f"""Вы — экспертный AI-преподаватель и аналитик. 
Используя приведенный ниже контекст из PDF-документа, дайте ПОДРОБНЫЙ и ДЕТАЛЬНЫЙ ответ на вопрос пользователя.

ТРЕБОВАНИЯ К ОТВЕТУ:
1. Подробно объясните суть темы простым и понятным языком.
2. Разбейте ответ на логические блоки с заголовками.
3. Выделите ключевые термины, формулы или факты.
4. Укажите, с каких конкретно страниц взята информация.
5. Если в контексте есть детали или цифры, обязательно приведите их.

ВОПРОС ПОЛЬЗОВАТЕЛЯ: {query}

КОНТЕКСТ ИЗ ДОКУМЕНТА:
{full_context}
"""

# Если введен API ключ Groq
                if groq_api_key.strip():
                    try:
                        from groq import Groq
                        
                        client = Groq(api_key=groq_api_key.strip())
                        chat_completion = client.chat.completions.create(
                            messages=[{"role": "user", "content": prompt}],
                            model="llama-3.3-70b-versatile",
                            temperature=0.3,
                            max_tokens=2048,
                        )
                        ai_response = chat_completion.choices[0].message.content
                        st.markdown("### 📚 Подробное объяснение от AI")
                        st.markdown(ai_response)
                    except ModuleNotFoundError:
                        st.error("Пакет 'groq' еще устанавливается на сервере. Подождите 10-15 секунд и попробуйте снова.")
                    except Exception as groq_err:
                        st.error(f"Ошибка при вызове Groq API: {str(groq_err)}")
                else:
                    # Режим без API-ключа: показываем извлеченный текст
                    st.success("✅ Релевантные страницы найдены и извлечены!")
                    st.markdown("### 📋 Извлеченный контекст из документа:")
                    st.info("Введите бесплатный Groq API Key в меню слева для автоматической генерации развернутого ответа через Llama 3.3.")
                    st.text_area("Скомпонованный контекст:", full_context, height=300)

                # Показываем исходные карточки
                st.subheader("📌 Источники (найденные страницы):")
                for idx, hit in enumerate(results):
                    payload = hit.payload or {}
                    with st.expander(f"Страница {payload.get('page')} (Score: {hit.score:.4f})"):
                        st.write(payload.get("text"))
            else:
                st.info("Ничего не найдено по данному запросу.")
        except Exception as e:
            st.error(f"Ошибка выполнения: {str(e)}")
