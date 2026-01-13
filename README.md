# Asistente Turístico con RAG + Memoria + Function Calling

## Requisitos
- Python 3.10+
- Variable de entorno `OPENAI_API_KEY` (opcional — si no existe se usa embedding fallback)

## Instalación
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Ejecución rápida
- Notebook: abre `notebook.ipynb` y ejecuta las celdas en orden (indexación, búsqueda y ejemplos).
- Streamlit (demo):
```bash
streamlit run app_streamlit.py
```

## Qué incluye
- RAG sobre `data/TENERIFE.pdf` con chunking y embeddings (usa OpenAI si `OPENAI_API_KEY` está definida, si no, usa un embedding de respaldo determinista).
- Indexación FAISS (si la biblioteca está disponible); el notebook ofrece un fallback por fuerza bruta.
- Diálogo multiturno: `src/rag_tools.ConversationMemory` con truncado básico por tokens.
- Function calling: `get_weather(fecha)` definida en `src/rag_tools.py` usando Pydantic y logueo en `logs/weather.log`.

## Notas de evaluación (para sacar un 10)
- Ejecuta y documenta los experimentos en `notebook.ipynb`.
- Asegúrate de: 1) usar variable de entorno para la API key, 2) mostrar parámetros del modelo (temperatura, max_tokens), 3) persistir el índice FAISS y recuperar citas de fuentes.
- Realiza al menos 3 invocaciones a `get_weather` (el notebook ya incluye 3 llamadas de ejemplo y registra en `logs/weather.log`).

Si quieres, puedo persistir el índice en `artifacts/` y añadir celdas para métricas y visualizaciones.
