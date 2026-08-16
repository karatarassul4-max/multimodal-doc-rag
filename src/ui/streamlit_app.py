import io
import base64
import tempfile
import streamlit as st
import fitz  # PyMuPDF
from pptx import Presentation
from PIL import Image
from gradio_client import Client

st.set_page_config(page_title="Streamlit + Gradio ZeroGPU LangGraph Explainer", layout="wide")

st.title("⚡ Streamlit + Hugging Face ZeroGPU LangGraph Explainer")

st.sidebar.header("⚙️ Settings")
space_name = st.sidebar.text_input("HF Space Name:", value="your-username/your-space-name")
hf_token = st.sidebar.text_input("Hugging Face Token (hf_...):", type="password")

uploaded_file = st.file_uploader("Загрузите PDF или PPTX документ", type=["pdf", "pptx"])
user_instruction = st.text_area("Фокус-указания для AI:", value="Детальный разбор всех схем и терминов.")

def image_to_base64(pil_image: Image.Image) -> str:
    buffered = io.BytesIO()
    pil_image.convert("RGB").save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

if st.button("🚀 Запустить обработку на ZeroGPU"):
    if not uploaded_file or not hf_token or not space_name:
        st.error("Заполните все поля!")
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

    item_label = "Страница" if file_ext == "pdf" else "Слайд"

    st.info("Отправка запроса на Hugging Face ZeroGPU Space...")

    try:
        with st.spinner("ZeroGPU A100 ускоритель выполняет LangGraph граф..."):
            client = Client(space_name, hf_token=hf_token)
            result = client.predict(
                hf_token=hf_token,
                user_instruction=user_instruction,
                detail_level="Глубокий",
                item_label=item_label,
                pages_base64=pages_b64,
                pages_text=pages_txt,
                api_name="/predict"
            )

            st.success("Готово! Данные успешно обработаны на Nvidia A100 ZeroGPU.")
            st.sidebar.metric("Оценка качества", f"{result['quality_score']}/10")
            st.sidebar.write(f"Фидбек критика: {result['critic_feedback']}")

            st.markdown("---")
            st.markdown("## 📸 Разбор страниц")
            for text in result["vision_analyses"]:
                st.markdown(text)

            st.markdown("---")
            st.markdown("## 🏛️ Итоговый отчет")
            st.markdown(result["final_output"])

    except Exception as e:
        st.error(f"Ошибка при вызове ZeroGPU: {str(e)}")
