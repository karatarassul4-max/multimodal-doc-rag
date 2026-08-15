FROM python:3.10-slim

WORKDIR /app

# Системные зависимости и Qdrant
RUN apt-get update && apt-get install -y \
    poppler-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Скачиваем бинарник Qdrant для запуска в том же контейнере
RUN curl -L https://github.com/qdrant/qdrant/releases/download/v1.8.0/qdrant-x86_64-unknown-linux-gnu.tar.gz | tar xz -C /usr/local/bin

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

# Скрипт запуска: стартует Qdrant, FastAPI и Streamlit одновременно
CMD qdrant & uvicorn src.api.app:app --host 0.0.0.0 --port 8000 & streamlit run src.ui.streamlit_app.py --server.port 8501 --server.address 0.0.0.0
