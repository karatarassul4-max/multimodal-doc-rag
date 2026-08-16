import io
import base64
import tempfile
import streamlit as st
import fitz  # PyMuPDF
from PIL import Image
from gradio_client import Client

st.set_page_config(page_title="Streamlit + Gradio ZeroGPU Explainer", layout="wide")

st.title("⚡ Document Explainer (Streamlit Client + ZeroGPU Backend)")

# --- Автоматическое считывание секретов ---
# Streamlit подтянет их из Secrets в Cloud или из .streamlit/secrets.toml локально
HF_SPACE_NAME = st.secrets.get("HF_SPACE_NAME", "")
HF_TOKEN = st.secrets.get("HF_TOKEN", "")

# Опционально: отобразить статус подключения в боковой панели
st.sidebar.header("⚙️ Статус подключения")
if HF_SPACE_NAME and HF_TOKEN:
    st.sidebar.success(f"Подключено к Space: `{HF_SPACE_NAME}`")
else:
    st.sidebar.error("Секреты HF_SPACE_NAME или HF_TOKEN не найдены в Secrets!")

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
        st.error("Ошибка конфигурации: Проверьте настройки Secrets в Streamlit!")
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

    st.info("Подключение к Hugging Face Space...")

    try:
        with st.spinner("Запуск обработки на ZeroGPU A100..."):
            # Заменили hf_token на token
            client = Client(HF_SPACE_NAME, token=HF_TOKEN)
            
            result = client.predict(
                hf_token=HF_TOKEN,
                user_instruction=user_instruction,
                detail_level="Глубокий",
                item_label="Страница",
                pages_base64=pages_b64,
                pages_text=pages_txt,
                api_name="/predict"
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
        st.error(f"Ошибка при вызове: {str(e)}")
