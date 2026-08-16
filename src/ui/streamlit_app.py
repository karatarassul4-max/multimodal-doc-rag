import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

import io
import base64
import tempfile
import concurrent.futures
import streamlit as st
import fitz  # PyMuPDF
from pptx import Presentation
from PIL import Image

st.set_page_config(page_title="Multimodal AI Visual & Deep Explainer", layout="wide")

st.title("👁️ Multimodal AI Explainer: Визуальный & Глубокий разбор PDF и PPTX")
st.write("Система конвертирует страницы/слайды в форматы высокого разрешения, анализирует схемы, таблицы и графики с помощью Vision LLM и формирует детализированный разбор.")

# --- Вспомогательные функции ---

def image_to_base64(pil_image: Image.Image) -> str:
    """Преобразует PIL Image в base64 строку для Groq Vision API."""
    buffered = io.BytesIO()
    pil_image.convert("RGB").save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def render_pdf_page_to_image(doc, page_num: int) -> Image.Image:
    """Рендерит страницу PDF в высокого разрешения PIL Image."""
    page = doc[page_num]
    pix = page.get_pixmap(dpi=150)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    return img


def render_pptx_slide_to_data(slide, slide_idx: int) -> tuple[str, list]:
    """Извлекает текст и встроенные изображения со слайда PPTX."""
    slide_text = []
    images = []
    for shape in slide.shapes:
        if hasattr(shape, "text") and shape.text.strip():
            slide_text.append(shape.text.strip())
        if shape.shape_type == 13:  # Picture shape type in python-pptx
            image_bytes = shape.image.blob
            img = Image.open(io.BytesIO(image_bytes))
            images.append(img)
    return "\n".join(slide_text), images


def analyze_page_with_vision(groq_client, image: Image.Image, text_content: str, item_type: str, item_num: int) -> str:
    """Анализирует изображение страницы/слайда с помощью Llama 3.2 Vision."""
    b64_img = image_to_base64(image)
    
    prompt = f"""Вы — главный эксперт по визуальному и текстовому анализу документов.
Перед вами {item_type} №{item_num}.

ИЗВЛЕЧЕННЫЙ ТЕКСТ:
{text_content if text_content else '[Текст отсутствует или содержится только на изображениях]'}

ИНСТРУКЦИЯ ПО АНАЛИЗУ:
1. Подробно опишите, что изображено на {item_type}: схемы, графики, таблицы, диаграммы, картинки или архитектуры.
2. Объясните взаимосвязь визуальных элементов с текстом.
3. Разжуйте ключевые тезисы, формулы, термины и метрики, присутствующие на этой странице.
4. Отвечайте детально и структурированно на русском языке.
"""
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}
                        }
                    ]
                }
            ],
            temperature=0.2,
            max_tokens=1500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ Ошибка Vision анализа {item_type} №{item_num}: {str(e)}"

# --- Настройки Сайдбара ---
st.sidebar.header("⚙️ Настройки AI Pipeline")
groq_api_key = st.sidebar.text_input(
    "Groq API Key (Обязательно):", 
    type="password", 
    help="Ключ с поддержкой Vision (Llama 3.2 Vision & Llama 3.3)"
)

detail_level = st.sidebar.select_slider(
    "Уровень детализации анализа:",
    options=["Стандартный", "Глубокий академический", "Экстремально подробный (Step-by-Step)"]
)

log_container = st.expander("🛠️ Pipeline Execution Trace & Logs", expanded=True)

def log_msg(msg: str, status: str = "info"):
    with log_container:
        if status == "success":
            st.success(f"[TRACE]: {msg}")
        elif status == "warning":
            st.warning(f"[TRACE]: {msg}")
        elif status == "error":
            st.error(f"[TRACE]: {msg}")
        else:
            st.info(f"[TRACE]: {msg}")

# --- Основной UI ---
uploaded_file = st.file_uploader("Загрузите PDF или PPTX документ", type=["pdf", "pptx"])
user_instruction = st.text_area(
    "Специальные фокус-указания для AI:", 
    value="Проведи детальный разбор. Разжуй все термины, таблицы, схемы, графики и скрытые смыслы. Объясни сложные вещи простым языком с примерами.",
    height=80
)

if st.button("🚀 Запустить Multimodal AI Analysis"):
    log_msg("Кнопка нажата. Старт обработки...", "info")

    if not uploaded_file:
        log_msg("Ошибка: Файл не загружен.", "error")
        st.error("Загрузите PDF или PPTX файл!")
        st.stop()

    if not groq_api_key.strip():
        log_msg("Ошибка: API Key отсутствует.", "error")
        st.error("Введите Groq API Key!")
        st.stop()

    from groq import Groq
    groq_client = Groq(api_key=groq_api_key.strip())

    file_ext = Path(uploaded_file.name).suffix.lower()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    pages_data = []  # List of tuples: (page_num, PIL_image, text)

    log_msg(f"Конвертация файла {uploaded_file.name} в растровые изображения...", "info")

    # --- Парсинг документов ---
    try:
        if file_ext == ".pdf":
            doc = fitz.open(tmp_path)
            log_msg(f"Всего страниц в PDF: {len(doc)}", "info")
            for page_idx in range(len(doc)):
                img = render_pdf_page_to_image(doc, page_idx)
                text = doc[page_idx].get_text("text")
                pages_data.append((page_idx + 1, img, text))
            doc.close()

        elif file_ext in [".pptx", ".ppt"]:
            prs = Presentation(tmp_path)
            log_msg(f"Всего слайдов в PPTX: {len(prs.slides)}", "info")
            for slide_idx, slide in enumerate(prs.slides):
                slide_text, slide_imgs = render_pptx_slide_to_data(slide, slide_idx)
                # Выбираем изображение для слайда
                if slide_imgs:
                    img = slide_imgs[0]
                else:
                    # Пустая заглушка-картинка с номером слайда, если нет встроенных картинок
                    img = Image.new('RGB', (800, 600), color=(240, 240, 240))
                pages_data.append((slide_idx + 1, img, slide_text))

        log_msg(f"Успешно обработано {len(pages_data)} страниц/слайдов.", "success")
    except Exception as parse_err:
        log_msg(f"Ошибка парсинга: {str(parse_err)}", "error")
        st.error(f"Ошибка обработки документа: {str(parse_err)}")
        st.stop()

    # --- Vision-анализ каждой страницы ---
    item_label = "Страница" if file_ext == ".pdf" else "Слайд"
    vision_analyses = []

    st.markdown("---")
    st.markdown("## 📸 Постраничный Vision-Анализ & Визуальная Привязка")

    progress_bar = st.progress(0)
    
    for idx, (num, img, text) in enumerate(pages_data):
        log_msg(f"Обработка Vision LLM ({item_label} {num}/{len(pages_data)})...", "info")
        
        col_img, col_analysis = st.columns([1, 1])
        
        with col_img:
            st.image(img, caption=f"{item_label} №{num}", use_container_width=True)
        
        with col_analysis:
            with st.spinner(f"Llama Vision анализирует {item_label.lower()} №{num}..."):
                analysis_text = analyze_page_with_vision(groq_client, img, text, item_label, num)
                st.markdown(f"### 🔍 Разбор {item_label.lower()}а №{num}")
                st.markdown(analysis_text)
                vision_analyses.append(f"=== {item_label} {num} ===\n{analysis_text}")

        progress_bar.progress((idx + 1) / len(pages_data))

    # --- Финальный сквозной синтез (Llama 3.3 70B) ---
    log_msg("Генерация единого итогового синтеза высокой детализации...", "info")
    
    combined_vision_context = "\n\n".join(vision_analyses)

    synthesis_prompt = f"""Вы — ведущий AI Архитектор и главный аналитик. 
Ваша задача — составить ИСЧЕРПЫВАЮЩИЙ, ГЛУБОКИЙ и ДЕТАЛИЗИРОВАННЫЙ разбор всего документа на основе проведенного постраничного мультимодального анализа.

ФОКУС-УКАЗАНИЯ ПОЛЬЗОВАТЕЛЯ:
{user_instruction}

УРОВЕНЬ ДЕТАЛИЗАЦИИ: {detail_level}

ПОСТРАНИЧНЫЙ АНАЛИЗ (ПОЛНЫЙ КОНТЕКСТ):
{combined_vision_context[:35000]}

ТРЕБОВАНИЯ К ФИНАЛЬНОМУ ОТВЕТУ:
1. **Общий глубинный обзор**: В чём главная суть и архитектура документа.
2. **Детальная декомпозиция схем и графиков**: Объедините данные со всех страниц и поясните все причинно-следственные связи.
3. **Глоссарий и терминология**: Выделите и простыми словами объясните абсолютно все ключевые понятия и сокращения.
4. **Практические выводы и чек-лист**: Что из этого следует и как применять знания.
5. Пишите максимально подробно, структурированно, без абстрактных формулировок.
"""

    try:
        with st.spinner("Генерация итогового синтеза через Llama 3.3 70B..."):
            completion = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": synthesis_prompt}],
                temperature=0.3,
                max_tokens=4000,
            )
            final_output = completion.choices[0].message.content
            log_msg("Финальный синтез успешно завершён!", "success")

            st.markdown("---")
            st.markdown("## 🏛️ Итоговый Сводный Анализ Документа")
            st.markdown(final_output)

    except Exception as synth_err:
        log_msg(f"Ошибка при формировании синтеза: {str(synth_err)}", "error")
        st.error(f"Ошибка глобального синтеза: {str(synth_err)}")
