import streamlit as st
from gradio_client import Client
import json

# Инициализация Gradio Client
client = Client("имя_вашего_space/vlm-backend")

pages_b64_list = [...] # Ваш список Base64 страниц
pages_text_list = [...] # Ваш список извлеченного текста

accumulated_context = ""
all_results = []

progress_bar = st.progress(0)
status_text = st.empty()

for i, b64_data in enumerate(pages_b64_list):
    page_num = i + 1
    total_pages = len(pages_b64_list)
    
    status_text.text(f"Обработка страницы {page_num} из {total_pages}...")
    
    txt_fallback = pages_text_list[i] if i < len(pages_text_list) else ""
    
    # Вызываем API для ОДНОЙ страницы
    job = client.submit(
        str(hf_token),
        user_instruction,
        page_num,
        total_pages,
        b64_data,
        txt_fallback,
        accumulated_context,
        api_name="/predict_page"
    )
    
    res = job.result(timeout=60)
    
    # Сохраняем анализ
    page_analysis = res.get("page_analysis", "")
    all_results.append(f"### Страница {page_num}\n{page_analysis}")
    
    # Обновляем контекст для СЛЕДУЮЩЕЙ страницы
    accumulated_context = res.get("updated_context", accumulated_context)
    
    # Отображаем результат страницы в Streamlit сразу
    with st.expander(f"Страница {page_num}", expanded=True):
        st.markdown(page_analysis)
        
    progress_bar.progress((i + 1) / total_pages)

status_text.text("Обработка всех страниц завершена!")
