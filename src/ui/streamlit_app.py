import os
import json
import base64
import io
import streamlit as st
from gradio_client import Client

# Попытки импорта библиотек для обработки парсинга файлов
try:
    import pypdf
except ImportError:
    pypdf = None

try:
    import docx
except ImportError:
    docx = None

try:
    import pptx
except ImportError:
    pptx = None

try:
    from PIL import Image
except ImportError:
    Image = None


st.set_page_config(
    page_title="Multimodal Document AI",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Multimodal Document AI & RAG")
st.write("Загрузите документы любого формата (PDF, DOCX, PPTX, TXT, Изображения) для анализа.")

# --- 1. Автоматическая загрузка секретов из Secrets / ENV ---
DEFAULT_SPACE = st.secrets.get("HF_SPACE_URL", os.getenv("HF_SPACE_URL", "RassulKaratayev/multimodal-doc-rag"))
DEFAULT_TOKEN = st.secrets.get("HF_TOKEN", os.getenv("HF_TOKEN", ""))

st.sidebar.header("Настройки подключения")

# Если секреты заданы, поля предзаполняются автоматически
hf_space_url = st.sidebar.text_input("HF Space URL / Name", value=DEFAULT_SPACE)
hf_token = st.sidebar.text_input("Hugging Face Token", value=DEFAULT_TOKEN, type="password")

st.sidebar.markdown("---")
detail_level = st.sidebar.selectbox("Уровень детализации", ["Стандартный", "Глубокий"], index=1)
item_label = st.sidebar.text_input("Метка элементов", value="Страница")

# --- 2. Ввод данных и обработка универсальных типов файлов ---
user_instruction = st.text_area("Инструкция / Запрос к документу", value="Проанализируй документ и извлеки ключевую информацию.")

uploaded_files = st.file_uploader(
    "Загрузите файлы любого формата (PDF, DOCX, DOC, PPTX, PPT, TXT, PNG, JPG, JPEG, WEBP)",
    type=["pdf", "docx", "doc", "pptx", "ppt", "txt", "png", "jpg", "jpeg", "webp"],
    accept_multiple_files=True
)


def extract_file_content(file):
    """Извлекает base64-представление и текстовое содержимое файла в зависимости от типа."""
    filename = file.name
    ext = filename.split(".")[-1].lower()
    file_bytes = file.read()
    
    b64_str = ""
    extracted_text = f"Файл: {filename}\n"

    # Изображения
    if ext in ["png", "jpg", "jpeg", "webp"]:
        b64_encoded = base64.b64encode(file_bytes).decode("utf-8")
        b64_str = f"data:image/{ext};base64,{b64_encoded}"
        extracted_text += "[Изображение загружено]"

    # PDF документы
    elif ext == "pdf":
        b64_encoded = base64.b64encode(file_bytes).decode("utf-8")
        b64_str = f"data:application/pdf;base64,{b64_encoded}"
        if pypdf:
            try:
                reader = pypdf.PdfReader(io.BytesIO(file_bytes))
                text_pages = [page.extract_text() or "" for page in reader.pages]
                extracted_text += "\n".join(text_pages)
            except Exception as e:
                extracted_text += f"[Не удалось извлечь текст из PDF: {e}]"
        else:
            extracted_text += "[Библиотека pypdf не установлена]"

    # Word документы (DOCX)
    elif ext == "docx":
        b64_encoded = base64.b64encode(file_bytes).decode("utf-8")
        b64_str = f"data:application/vnd.openxmlformats-officedocument.wordprocessingml.document;base64,{b64_encoded}"
        if docx:
            try:
                doc = docx.Document(io.BytesIO(file_bytes))
                full_text = [p.text for p in doc.paragraphs if p.text]
                extracted_text += "\n".join(full_text)
            except Exception as e:
                extracted_text += f"[Ошибка чтения DOCX: {e}]"
        else:
            extracted_text += "[Библиотека python-docx не установлена]"

    # PowerPoint презентации (PPTX)
    elif ext == "pptx":
        b64_encoded = base64.b64encode(file_bytes).decode("utf-8")
        b64_str = f"data:application/vnd.openxmlformats-officedocument.presentationml.presentation;base64,{b64_encoded}"
        if pptx:
            try:
                prs = pptx.Presentation(io.BytesIO(file_bytes))
                slide_texts = []
                for idx, slide in enumerate(prs.slides):
                    slide_str = f"--- Слайд {idx+1} ---\n"
                    for shape in slide.shapes:
                        if hasattr(shape, "text"):
                            slide_str += shape.text + "\n"
                    slide_texts.append(slide_str)
                extracted_text += "\n".join(slide_texts)
            except Exception as e:
                extracted_text += f"[Ошибка чтения PPTX: {e}]"
        else:
            extracted_text += "[Библиотека python-pptx не установлена]"

    # Текстовые файлы (TXT)
    elif ext == "txt":
        try:
            text_content = file_bytes.decode("utf-8", errors="ignore")
            extracted_text += text_content
            b64_encoded = base64.b64encode(file_bytes).decode("utf-8")
            b64_str = f"data:text/plain;base64,{b64_encoded}"
        except Exception as e:
            extracted_text += f"[Ошибка чтения TXT: {e}]"

    # Остальные форматы (DOC, PPT)
    else:
        b64_encoded = base64.b64encode(file_bytes).decode("utf-8")
        b64_str = f"data:application/octet-stream;base64,{b64_encoded}"
        extracted_text += f"[Файл формата .{ext} передан в бинарном виде]"

    return b64_str, extracted_text


if st.button("🚀 Запустить обработку", type="primary"):
    if not hf_token:
        st.error("Пожалуйста, укажите Hugging Face Token (в Secrets или боковой панели).")
        st.stop()
        
    if not uploaded_files:
        st.warning("Загрузите хотя бы один файл для обработки.")
        st.stop()

    with st.spinner("Извлечение содержимого и кодирование файлов..."):
        pages_b64 = []
        pages_txt = []
        
        for file in uploaded_files:
            b64_data, text_data = extract_file_content(file)
            pages_b64.append(b64_data)
            pages_txt.append(text_data)

        # Сериализация списков в JSON-строки
        pages_b64_json = json.dumps(pages_b64)
        pages_txt_json = json.dumps(pages_txt)

    st.info(f"Обработано файлов: {len(pages_b64)}. Подключение к HF Space...")

    try:
        with st.spinner("Выполнение LangGraph пайплайна на ZeroGPU..."):
            # Исправленная инициализация клиента для всех версий gradio_client
            try:
                client = Client(hf_space_url, token=hf_token)
            except TypeError:
                client = Client(hf_space_url, hf_token=hf_token)
            
            result = client.predict(
                str(hf_token),
                str(user_instruction),
                str(detail_level),
                str(item_label),
                pages_b64_json,
                pages_txt_json,
                api_name="/predict"
            )

        st.success("Обработка завершена успешно!")
        
        # Отображение результатов
        if isinstance(result, dict):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Оценка качества (Quality Score)", result.get("quality_score", 0))
            with col2:
                st.metric("Повторные попытки (Retries)", result.get("retry_count", 0))
                
            st.subheader("Финал вывода")
            st.markdown(result.get("final_output", "Нет ответа"))
            
            st.subheader("Обратная связь критика")
            st.info(result.get("critic_feedback", "Нет отзывов"))
            
            with st.expander("Постраничный / Пофайловый анализ"):
                for idx, analysis in enumerate(result.get("vision_analyses", [])):
                    st.write(f"**Элемент {idx + 1}:** {analysis}")
        else:
            st.json(result)

    except Exception as e:
        st.error(f"Ошибка при вызове backend API: {str(e)}")
        st.exception(e)
