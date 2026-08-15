import streamlit as st
import requests

st.set_page_config(page_title="Multimodal Document RAG", layout="wide")

st.title("📄 Multimodal Document AI & RAG Search")
st.write("Загружайте PDF-документы и ищите информацию по тексту и изображениям.")

# Настройки API
API_URL = "http://localhost:8000"

st.sidebar.header("1. Загрузка документа")
uploaded_file = st.sidebar.file_uploader("Выберите PDF файл", type=["pdf"])

if uploaded_file is not None:
    if st.sidebar.button("Индексировать PDF"):
        with st.spinner("Парсинг и векторизация документа..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            response = requests.post(f"{API_URL}/upload_pdf", files=files)
            
            if response.status_code == 200:
                result = response.json()
                st.sidebar.success(f"Успешно! Заиндексировано страниц: {result['indexed_pages']}")
            else:
                st.sidebar.error("Ошибка при обработке файла.")

st.header("2. Мультимодальный поиск")
query = st.text_input("Введите поисковый запрос (например: 'график продаж за 3 квартал' или 'схема архитектуры'):")

if st.button("Найти в документах") and query:
    with st.spinner("Поиск по векторной базе..."):
        response = requests.post(f"{API_URL}/search", json={"query": query, "top_k": 3})
        
        if response.status_code == 200:
            data = response.json()
            st.subheader(f"Результаты по запросу: '{data['query']}'")
            
            for idx, res in enumerate(data["results"]):
                with st.expander(f"Результат #{idx+1} | Файл: {res['filename']} (Стр. {res['page']}) | Relevance Score: {res['score']:.4f}"):
                    st.write("**Извлеченный фрагмент:**")
                    st.write(res["text"])
        else:
            st.error("Ошибка выполнения поиска.")
