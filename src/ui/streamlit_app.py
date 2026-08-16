import io
import base64
import tempfile
import streamlit as st
import fitz  # PyMuPDF
from PIL import Image
from gradio_client import Client

st.set_page_config(page_title="Streamlit + Gradio ZeroGPU Explainer", layout="wide")

st.title("⚡ Document Explainer (Streamlit Client + ZeroGPU Backend)")

st.sidebar.header("⚙️ Конфигурация")
space_name = st.sidebar.text_input("HF Space Name (username/space-name):")
hf_token = st.sidebar.text_input("Hugging Face Token:", type="password")

uploaded_file = st.file_uploader("Загрузите PDF файл", type=["pdf"])
user_instruction = st.text_area("Фокус-указания для AI:", value="Подробный разбор всех графиков, схем и текста.")

def image_to_base64(pil_image: Image.Image) -> str:
    buffered = io.BytesIO()
    pil_image.convert("RGB").save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

if st.button("🚀 Отправить на ZeroGPU"):
    if not uploaded_file or not hf_token or not space_name:
        st.error("Заполните HF Space Name, HF Token и загрузите файл!")
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
            client = Client(space_name, hf_token=hf_token)
            result = client.predict(
                hf_token=hf_token,
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
