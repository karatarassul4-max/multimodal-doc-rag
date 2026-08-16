import os
import json
import base64
import streamlit as st
from gradio_client import Client

st.set_page_config(
    page_title="Multimodal Document AI",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Multimodal Document AI & RAG")
st.write("Загрузите документ или изображения и отправьте их на обработку в Hugging Face ZeroGPU backend.")

# Sidebar с настройками
st.sidebar.header("Настройки подключения")
hf_space_url = st.sidebar.text_input("HF Space URL / Name", value="your-username/your-space-name")
hf_token = st.sidebar.text_input("Hugging Face Token", type="password")

st.sidebar.markdown("---")
detail_level = st.sidebar.selectbox("Уровень детализации", ["Стандартный", "Глубокий"], index=1)
item_label = st.sidebar.text_input("Метка элементов", value="Страница")

# Основной интерфейс
user_instruction = st.text_area("Инструкция / Запрос к документу", value="Проанализируй документ и извлеки ключевую информацию.")

uploaded_files = st.file_uploader(
    "Загрузите страницы документа (PNG, JPG, JPEG)",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True
)

if st.button("🚀 Запустить обработку", type="primary"):
    if not hf_token:
        st.error("Пожалуйста, укажите Hugging Face Token в боковой панели.")
        st.stop()
        
    if not uploaded_files:
        st.warning("Загрузите хотя бы один файл для обработки.")
        st.stop()

    with st.spinner("Подготовка файлов и кодирование в Base64..."):
        pages_b64 = []
        pages_txt = []
        
        for file in uploaded_files:
            file_bytes = file.read()
            b64_encoded = base64.b64encode(file_bytes).decode("utf-8")
            pages_b64.append(f"data:{file.type};base64,{b64_encoded}")
            pages_txt.append(file.name)

        # Сериализация списков в JSON-строки
        pages_b64_json = json.dumps(pages_b64)
        pages_txt_json = json.dumps(pages_txt)

    st.info(f"Подготовлено страниц: {len(pages_b64)}. Подключение к HF Space...")

    try:
        with st.spinner("Выполнение LangGraph пайплайна на ZeroGPU..."):
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
            
            with st.expander("Постраничный анализ"):
                for idx, analysis in enumerate(result.get("vision_analyses", [])):
                    st.write(f"**Страница {idx + 1}:** {analysis}")
        else:
            st.json(result)

    except Exception as e:
        st.error(f"Ошибка при вызове backend API: {str(e)}")
        st.exception(e)
