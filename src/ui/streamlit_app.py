import io
import base64
import tempfile
import traceback
import streamlit as st
import fitz  # PyMuPDF
from PIL import Image
from gradio_client import Client

st.set_page_config(page_title="Streamlit + Gradio ZeroGPU Explainer", layout="wide")

st.title("⚡ Document Explainer (Streamlit Client + ZeroGPU Backend)")

# --- Инициализация системных логов в session_state ---
if "logs" not in st.session_state:
    st.session_state.logs = []

def log(msg: str):
    st.session_state.logs.append(msg)

# --- Чтение секретов ---
HF_SPACE_NAME = st.secrets.get("HF_SPACE_NAME", "karatarassul4/langgraph-vision-explainer")
HF_TOKEN = st.secrets.get("HF_TOKEN", "")

# Боковая панель: статус и логи
st.sidebar.header("⚙️ Конфигурация")
if HF_SPACE_NAME and HF_TOKEN:
    st.sidebar.success(f"Connected: `{HF_SPACE_NAME}`")
else:
    st.sidebar.error("Проверьте Secrets в Streamlit Cloud!")

with st.sidebar.expander("📋 Логи работы (Real-time)", expanded=True):
    if st.button("Очистить логи"):
        st.session_state.logs = []
        st.rerun()
    log_box = st.empty()
    log_box.code("\n".join(st.session_state.logs) if st.session_state.logs else "Ожидание запуска...", language="text")

uploaded_file = st.file_uploader("Загрузите PDF файл", type=["pdf"])
user_instruction = st.text_area("Фокус-указания для AI:", value="Подробный разбор всех графиков, схем и текста.")

def image_to_base64(pil_image: Image.Image) -> str:
    buffered = io.BytesIO()
    pil_image.convert("RGB").save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

if st.button("🚀 Отправить на ZeroGPU"):
    if not uploaded_file:
        st.error("Загрузите PDF файл!")
        st.stop()
        
    log(f"Начало обработки файла: {uploaded_file.name}")
    log_box.code("\n".join(st.session_state.logs), language="text")

    file_ext = uploaded_file.name.split(".")[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    pages_b64 = []
    pages_txt = []

    if file_ext == "pdf":
        doc = fitz.open(tmp_path)
        log(f"Извлечение страниц из PDF (Всего страниц: {len(doc)})...")
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            pix = page.get_pixmap(dpi=150)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            pages_b64.append(image_to_base64(img))
            pages_txt.append(page.get_text("text"))
        doc.close()

    log(f"Подготовка payload: {len(pages_b64)} изображений base64...")
    log_box.code("\n".join(st.session_state.logs), language="text")

    try:
        with st.spinner("Запуск обработки на ZeroGPU..."):
            log(f"Инициализация Gradio Client для {HF_SPACE_NAME}...")
            client = Client(HF_SPACE_NAME, token=HF_TOKEN)

            log("Отправка запроса в Gradio API...")
            log_box.code("\n".join(st.session_state.logs), language="text")

            # Передача именованных аргументов
            result = client.predict(
                hf_token=HF_TOKEN,
                user_instruction=user_instruction,
                detail_level="Глубокий",
                item_label="Страница",
                pages_base64=pages_b64,
                pages_text=pages_txt,
                api_name="/predict"
            )

            log("Запрос успешно выполнен!")
            log_box.code("\n".join(st.session_state.logs), language="text")

            st.success("Обработка завершена успешно!")
            st.sidebar.metric("Оценка качества", f"{result['quality_score']}/10")
            st.sidebar.write(f"Фидбек: {result['critic_feedback']}")

            st.markdown("---")
            st.markdown("## 📸 Разбор страниц")
            for text in result.get("vision_analyses", []):
                st.markdown(text)

            st.markdown("---")
            st.markdown("## 🏛️ Итоговый отчет")
            st.markdown(result.get("final_output", ""))

    except Exception as e:
        err_msg = f"[ERROR] {type(e).__name__}: {str(e)}"
        log(err_msg)
        log_box.code("\n".join(st.session_state.logs), language="text")
        
        st.error(f"**Ошибка вызова API:** `{type(e).__name__}: {str(e)}`")
        with st.expander("🔍 Подробный Traceback"):
            st.code(traceback.format_exc())
