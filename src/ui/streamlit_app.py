import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

import io
import base64
import tempfile
import streamlit as st
import fitz  # PyMuPDF
from pptx import Presentation
from PIL import Image

# --- LangChain Imports ---
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="LangChain Multimodal Visual Explainer", layout="wide")

st.title("🦜🔗 LangChain Vision Explainer: Глубокий разбор PDF и PPTX")
st.write("Пайплайн на базе **LangChain (LCEL)**, который берет визуальные страницы документов, пропускает их через Vision LLM и генерирует исчерпывающий разбор.")

# --- Вспомогательные функции ---

def image_to_base64(pil_image: Image.Image) -> str:
    """Преобразует PIL Image в base64 строку для Vision API."""
    buffered = io.BytesIO()
    pil_image.convert("RGB").save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def render_pdf_page_to_image(doc, page_num: int) -> Image.Image:
    """Рендерит страницу PDF в PIL Image."""
    page = doc[page_num]
    pix = page.get_pixmap(dpi=150)
    return Image.open(io.BytesIO(pix.tobytes("png")))


def render_pptx_slide_to_data(slide) -> tuple[str, list]:
    """Извлекает текст и картинки со слайда PPTX."""
    slide_text = []
    images = []
    for shape in slide.shapes:
        if hasattr(shape, "text") and shape.text.strip():
            slide_text.append(shape.text.strip())
        if shape.shape_type == 13:
            img = Image.open(io.BytesIO(shape.image.blob))
            images.append(img)
    return "\n".join(slide_text), images


def run_langchain_vision_analysis(api_key: str, image: Image.Image, text_content: str, item_type: str, item_num: int) -> str:
    """Выполняет Vision-анализ страницы с использованием LangChain ChatGroq."""
    b64_img = image_to_base64(image)
    
    # Инициализация модели через LangChain
    vision_model = ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.2-11b-vision-preview",
        temperature=0.2,
        max_tokens=1500
    )

    prompt_text = f"""Перед вами {item_type} №{item_num}.

ИЗВЛЕЧЕННЫЙ ТЕКСТ:
{text_content if text_content else '[Текст отсутствует или содержится на изображениях]'}

ИНСТРУКЦИЯ ПО АНАЛИЗУ:
1. Подробно опишите, что изображено на {item_type}: схемы, графики, таблицы, диаграммы или архитектуры.
2. Объясните взаимосвязь визуальных элементов с текстом.
3. Разжуйте ключевые тезисы, формулы, термины и метрики.
4. Отвечайте детально и структурированно на русском языке.
"""

    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt_text},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}
            }
        ]
    )

    # LangChain LCEL Execution Pipeline
    chain = vision_model | StrOutputParser()
    return chain.invoke([message])


def run_langchain_synthesis(api_key: str, user_instruction: str, detail_level: str, combined_context: str) -> str:
    """Генерирует сводный аналитический разбор через LangChain LCEL Chain."""
    
    synthesis_model = ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0.3,
        max_tokens=4000
    )

    # LangChain ChatPromptTemplate
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "Вы — ведущий AI Архитектор и главный аналитик. Составляйте исчерпывающие, глубокие и структурированные разборы материалов."),
        ("human", """ФОКУС-УКАЗАНИЯ ПОЛЬЗОВАТЕЛЯ:
{user_instruction}

УРОВЕНЬ ДЕТАЛИЗАЦИИ: {detail_level}

ПОСТРАНИЧНЫЙ ВКУПНОЙ КОНТЕКСТ:
{context}

ТРЕБОВАНИЯ К ФИНАЛЬНОМУ ОТВЕТУ:
1. **Общий глубинный обзор**: В чём главная суть и архитектура документа.
2. **Детальная декомпозиция схем и графиков**: Объедините данные со всех страниц и поясните все причинно-следственные связи.
3. **Глоссарий и терминология**: Выделите и простыми словами объясните абсолютно все ключевые понятия и сокращения.
4. **Практические выводы и чек-лист**: Что из этого следует и как применять знания.
5. Пишите максимально подробно и структурированно.
""")
    ])

    # Сборка цепочки через LCEL (Prompt | Model | OutputParser)
    chain = prompt_template | synthesis_model | StrOutputParser()

    return chain.invoke({
        "user_instruction": user_instruction,
        "detail_level": detail_level,
        "context": combined_context[:35000]
    })

# --- Настройки Сайдбара ---
st.sidebar.header("⚙️ LangChain Pipeline Config")
groq_api_key = st.sidebar.text_input(
    "Groq API Key (Обязательно):", 
    type="password", 
    help="Ключ от console.groq.com"
)

detail_level = st.sidebar.select_slider(
    "Уровень детализации анализа:",
    options=["Стандартный", "Глубокий академический", "Экстремально подробный (Step-by-Step)"]
)

log_container = st.expander("🛠️ LangChain Execution Trace Logs", expanded=True)

def log_msg(msg: str, status: str = "info"):
    with log_container:
        if status == "success":
            st.success(f"[LANGCHAIN TRACE]: {msg}")
        elif status == "warning":
            st.warning(f"[LANGCHAIN TRACE]: {msg}")
        elif status == "error":
            st.error(f"[LANGCHAIN TRACE]: {msg}")
        else:
            st.info(f"[LANGCHAIN TRACE]: {msg}")

# --- UI ---
uploaded_file = st.file_uploader("Загрузите PDF или PPTX документ", type=["pdf", "pptx"])
user_instruction = st.text_area(
    "Специальные фокус-указания для AI:", 
    value="Проведи детальный разбор. Разжуй все термины, таблицы, схемы, графики и скрытые смыслы. Объясни сложные вещи простым языком с примерами.",
    height=80
)

if st.button("🚀 Запустить LangChain Multimodal Pipeline"):
    log_msg("Инициализация LangChain Pipeline...", "info")

    if not uploaded_file:
        log_msg("Ошибка: Файл не загружен.", "error")
        st.error("Загрузите PDF или PPTX файл!")
        st.stop()

    if not groq_api_key.strip():
        log_msg("Ошибка: API Key отсутствует.", "error")
        st.error("Введите Groq API Key!")
        st.stop()

    file_ext = Path(uploaded_file.name).suffix.lower()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    pages_data = []

    log_msg(f"Конвертация файла {uploaded_file.name}...", "info")

    try:
        if file_ext == ".pdf":
            doc = fitz.open(tmp_path)
            for page_idx in range(len(doc)):
                img = render_pdf_page_to_image(doc, page_idx)
                text = doc[page_idx].get_text("text")
                pages_data.append((page_idx + 1, img, text))
            doc.close()

        elif file_ext in [".pptx", ".ppt"]:
            prs = Presentation(tmp_path)
            for slide_idx, slide in enumerate(prs.slides):
                slide_text, slide_imgs = render_pptx_slide_to_data(slide)
                img = slide_imgs[0] if slide_imgs else Image.new('RGB', (800, 600), color=(240, 240, 240))
                pages_data.append((slide_idx + 1, img, slide_text))

        log_msg(f"Успешно подгружено {len(pages_data)} страниц/слайдов.", "success")
    except Exception as parse_err:
        log_msg(f"Ошибка парсинга: {str(parse_err)}", "error")
        st.error(f"Ошибка обработки: {str(parse_err)}")
        st.stop()

    # --- LangChain Vision Chain execution ---
    item_label = "Страница" if file_ext == ".pdf" else "Слайд"
    vision_analyses = []

    st.markdown("---")
    st.markdown("## 📸 Постраничный LangChain Vision-Анализ")

    progress_bar = st.progress(0)
    
    for idx, (num, img, text) in enumerate(pages_data):
        log_msg(f"Запуск LangChain Vision Chain ({item_label} {num}/{len(pages_data)})...", "info")
        
        col_img, col_analysis = st.columns([1, 1])
        
        with col_img:
            st.image(img, caption=f"{item_label} №{num}", use_container_width=True)
        
        with col_analysis:
            with st.spinner(f"LangChain ChatGroq Vision анализирует {item_label.lower()} №{num}..."):
                try:
                    analysis_text = run_langchain_vision_analysis(groq_api_key.strip(), img, text, item_label, num)
                    st.markdown(f"### 🔍 Разбор {item_label.lower()}а №{num}")
                    st.markdown(analysis_text)
                    vision_analyses.append(f"=== {item_label} {num} ===\n{analysis_text}")
                except Exception as ve:
                    log_msg(f"Ошибка Vision Chain на странице {num}: {str(ve)}", "error")
                    st.error(f"Ошибка анализа страницы {num}: {str(ve)}")

        progress_bar.progress((idx + 1) / len(pages_data))

    # --- LangChain Final Synthesis Chain ---
    log_msg("Запуск LangChain Synthesis Chain (ChatPromptTemplate | ChatGroq | StrOutputParser)...", "info")
    
    combined_vision_context = "\n\n".join(vision_analyses)

    try:
        with st.spinner("LangChain генерирует итоговый синтез через Llama 3.3 70B..."):
            final_output = run_langchain_synthesis(
                groq_api_key.strip(),
                user_instruction,
                detail_level,
                combined_vision_context
            )
            log_msg("LangChain Synthesis Chain успешно завершила работу!", "success")

            st.markdown("---")
            st.markdown("## 🏛️ Итоговый Сводный Анализ (LangChain Output)")
            st.markdown(final_output)

    except Exception as synth_err:
        log_msg(f"Ошибка в Synthesis Chain: {str(synth_err)}", "error")
        st.error(f"Ошибка глобального синтеза: {str(synth_err)}")
