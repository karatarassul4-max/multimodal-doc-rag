import io
import base64
import logging
import tempfile
import traceback
import streamlit as st
import fitz  # PyMuPDF
from PIL import Image
from gradio_client import Client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Streamlit_Client")

st.set_page_config(page_title="Streamlit + Gradio ZeroGPU Explainer", layout="wide")

st.title("⚡ Document Explainer (Streamlit Client + ZeroGPU Backend)")

# --- Чтение секретов ---
HF_SPACE_NAME = st.secrets.get("HF_SPACE_NAME", "karatarassul4/langgraph-vision-explainer")
HF_TOKEN = st.secrets.get("HF_TOKEN", "")

st.sidebar.header("⚙️ Статус подключения")
if HF_SPACE_NAME and HF_TOKEN:
    st.sidebar.success(f"Space: `{HF_SPACE_NAME}`")
else:
    st.sidebar.error("HF_SPACE_NAME или HF_TOKEN не найдены в Secrets!")

uploaded_file = st.file_uploader("Загрузите PDF файл", type=["pdf"])
user_instruction = st.text_area("Фокус-указания для AI:", value="Подробный разбор всех графиков, схем и текста.")

def image_to_base64(pil_image: Image.Image) -> str:
    buffered = io.BytesIO()
    pil_image.convert("RGB").save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

if st.button("🚀 Отправить на ZeroGPU"):
    if not uploaded_file:
        st.error("Пожалуйста, загрузите PDF файл!")
        st.stop()
        
    if not HF_SPACE_NAME or not HF_TOKEN:
        st.error("Ошибка конфигурации: Проверьте настройки Secrets!")
        st.stop()

    file_ext = uploaded_file.name.split(".")[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    pages_b64 = []
    pages_txt = []

    if file_ext == "pdf":
        doc = fitz.open(tmp_path)
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            pix = page.get_pixmap(dpi=150)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            pages_b64.append(image_to_base64(img))
            pages_txt.append(page.get_text("text"))
        doc.close()

    logger.info(f"Извлечено страниц из PDF: {len(pages_b64)}")

    try:
        with st.spinner("Запуск обработки на ZeroGPU..."):
            logger.info(f"Инициализация Client('{HF_SPACE_NAME}')...")
            client = Client(HF_SPACE_NAME, token=HF_TOKEN)
            
            logger.info("Отправка вызова client.predict()...")
            result = client.predict(
                HF_TOKEN,
                user_instruction,
                "Глубокий",
                "Страница",
                pages_b64,
                pages_txt
            )

            st.success("Обработка завершена успешно!")
            st.sidebar.metric("Оценка качества", f"{result['quality_score']}/10")
            st.sidebar.write(f"Фидбек: {result['critic_feedback']}")

            st.markdown("---")
            st.markdown("## 📸 Разбор страниц")
            for text in result["vision_analyses"]:
                st.markdown(text)

            st.markdown("---")
            st.markdown("## 🏛️ Итоговый отчет")
            st.markdown(result["final_output"])

    except Exception as e:
        logger.error(f"Ошибка при вызове: {type(e).__name__} - {str(e)}")
        logger.error(traceback.format_exc())
        st.error(f"**Ошибка вызова API:** `{type(e).__name__}: {str(e)}`")
        with st.expander("🔍 Подробности Traceback"):
            st.code(traceback.format_exc())
