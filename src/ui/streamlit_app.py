if st.button("Найти в документах") and query:
    with st.spinner("Поиск по векторной базе..."):
        try:
            query_vector = encoder.encode_text(query)

            # Используем актуальный метод query_points вместо устаревшего search
            search_result = vector_store.client.query_points(
                collection_name="multimodal_docs",
                query=query_vector,
                limit=3
            )

            st.subheader(f"Результаты по запросу: '{query}'")

            # Извлекаем точки из ответа query_points
            results = search_result.points

            if results:
                for idx, hit in enumerate(results):
                    payload = hit.payload or {}
                    filename = payload.get("filename", "Unknown")
                    page = payload.get("page", "-")
                    text = payload.get("text", "")

                    with st.expander(
                        f"Результат #{idx+1} | Файл: {filename} (Стр. {page}) | Score: {hit.score:.4f}"
                    ):
                        st.write("**Извлеченный фрагмент:**")
                        st.write(text)
            else:
                st.info("Ничего не найдено.")
        except Exception as e:
            st.error(f"Ошибка при выполнении поиска: {str(e)}")
