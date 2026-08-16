import sys
from pathlib import Path

# Добавляем корень проекта в sys.path для корректных импортов
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

import tempfile
import streamlit as st
import fitz  # PyMuPDF для PDF
from pptx import Presentation  # python-pptx для PPTX

st.set_page_config(page_title="AI Document & Presentation Explainer", layout="wide")

st.title("📄 AI Explainer: Детальный разбор PDF и PPTX")
st.write("Загрузите презентацию или PDF-документ, чтобы AI подробно разжевал всю информацию, термины и выводы.")

# --- Функция извлечения текста ---
def extract_text_from_file(uploaded_file) -> tuple[str, int]:
    """Извлекает весь текст из PDF или PPTX файла."""
    file_ext = Path(uploaded_file.name).suffix.lower()
    full_text = []
    page_count = 0

    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    if file_ext == ".pdf":
        doc = fitz.open(tmp_path)
        page_count = len(doc)
        for page_num in range(page_count):
            text = doc[page_num].get_text("text")
            if text.strip():
                full_text.append(f"--- Страница {page_num + 1} ---\n{text.strip()}")
        doc.close()

    elif file_ext in [".pptx", ".ppt"]:
        prs = Presentation(tmp_path)
        page_count = len(prs.slides)
        for slide_idx, slide in enumerate(prs.slides):
            slide_text = []
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    slide_text.append(shape.text.strip())
            if slide_text:
                full_text.append(f"--- Слайд {slide_idx + 1} ---\n" + "\n".join(slide_text))

    return "\n\n".join(full_text), page_count

# --- Сайдбар с настройками ---
st.sidebar.header("⚙️ Настройки")
groq_api_key = st.sidebar.text_input(
    "Groq API Key (обязательно):", 
    type="password", 
    help="Бесплатный ключ от console.groq.com"
)

# Контейнер для отображения логов процесса
log_container = st.expander("🛠️ Логи выполнения (Debug Info)", expanded=True)

def log_msg(msg: str, status: str = "info"):
    with log_container:
        if status == "success":
            st.success(f"[LOG]: {msg}")
        elif status == "warning":
            st.warning(f"[LOG]: {msg}")
        elif status == "error":
            st.error(f"[LOG]: {msg}")
        else:
            st.info(f"[LOG]: {msg}")

# --- Загрузка файла ---
uploaded_file = st.file_uploader("Загрузите PDF или PPTX файл", type=["pdf", "pptx"])

user_instruction = st.text_area(
    "Что именно нужно разжевать?", 
    value="Подробно объясни ключевые темы, термины, цифры и выводы из этого файла простым языком.",
    height=100
)

# --- Кнопка генерации ---
if st.button("🚀 Объяснить подробно (AI Analysis)"):
    log_msg("Кнопка нажата!", "info")

    if not uploaded_file:
        log_msg("Ошибка: Файл не загружен.", "error")
        st.error("Пожалуйста, загрузите PDF или PPTX файл.")
        st.stop()

    if not groq_api_key.strip():
        log_msg("Ошибка: API ключ Groq не введен.", "error")
        st.error("Пожалуйста, введите Groq API Key в меню слева.")
        st.stop()

    # 1. Извлечение текста
    log_msg(f"Начинаем извлечение текста из файла: {uploaded_file.name}", "info")
    try:
        extracted_text, total_pages = extract_text_from_file(uploaded_file)
        log_msg(f"Текст успешно извлечен! Обработано страниц/слайдов: {total_pages}. Длина текста: {len(extracted_text)} символов.", "success")
    except Exception as e:
        log_msg(f"Ошибка при парсинге файла: {str(e)}", "error")
        st.error(f"Не удалось прочитать файл: {str(e)}")
        st.stop()

    if not extracted_text.strip():
        log_msg("Ошибка: Файл пуст или содержит только несчитываемые изображения.", "warning")
        st.warning("В файле не найден текст для анализа.")
        st.stop()

    # 2. Подготовка промпта
    log_msg("Формируем промпт для Llama 3.3...", "info")
    # Ограничиваем объём текста, если документ слишком огромный (чтобы влез в контекст)
    truncated_text = extracted_text[:30000]

    prompt = f"""Вы — экспертный аналитик и преподаватель. Ваши ответы всегда глубокие, структурированные и легко усваиваемые.

ЗАДАЧА ПОЛЬЗОВАТЕЛЯ:
{user_instruction}

СОДЕРЖИМОЕ ДОКУМЕНТА:
{truncated_text}

ИНСТРУКЦИЯ К ОТВЕТУ:
1. Дайте подробное и развернутое объяснение материала.
2. Используйте чёткие логические заголовки, списки и выделения ключевых терминов.
3. Поясните сложные понятия и приведите практические примеры или выводы на основе документа.
4. Отвечайте на русском языке.
"""

    # 3. Вызов Groq API
    log_msg("Отправка запроса в Groq API (модель llama-3.3-70b-versatile)...", "info")
    try:
        from groq import Groq
        client = Groq(api_key=groq_api_key.strip())
        
        with st.spinner("LLM генерирует подробный разбор материала..."):
            completion = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile",
                temperature=0.3,
                max_tokens=4096,
            )
            
            ai_output = completion.choices[0].message.content
            log_msg("Ответ от Groq API успешно получен!", "success")

            st.markdown("---")
            st.markdown("### 📚 Подробный разбор от AI:")
            st.markdown(ai_output)

            with st.expander("📄 Исходный текст из файла"):
                st.text_area("Извлеченный текст:", extracted_text, height=300)

    except ModuleNotFoundError:
        log_msg("Пакет 'groq' еще не установлен на сервере Streamlit.", "error")
        st.error("Пакет 'groq' еще устанавливается на сервере. Подождите 20 секунд и нажмите кнопку снова.")
    except Exception as api_err:
        log_msg(f"Ошибка при вызове Groq API: {str(api_err)}", "error")
        st.error(f"Ошибка вызова LLM: {str(api_err)}")
