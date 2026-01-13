# Informe Final

## Resumen
Se implementa un prototipo reproducible de asistente turístico con RAG, memoria conversacional y llamadas a funciones. El proyecto incluye:

- `src/rag_tools.py`: extracción de PDF, chunking, embeddings (OpenAI o fallback), wrappers FAISS, `get_weather` con Pydantic y logging, y memoria conversacional simple.
- `notebook.ipynb`: flujo reproducible: extracción de `data/TENERIFE.pdf`, chunking, embeddings, indexación, búsquedas y 3 llamadas a `get_weather`.
- `app_streamlit.py`: demo mínima para carga rápida, búsqueda y llamada a `get_weather`.

## Decisiones técnicas
- Embeddings: si `OPENAI_API_KEY` está definida, el notebook/intenciones usan OpenAI embeddings; en entornos sin key se usa un embedding fallback determinista basado en hashing de tokens para garantizar reproducibilidad.
- Index: FAISS se usa cuando está disponible; el notebook incluye un fallback por fuerza bruta (cosine) para entornos sin FAISS.
- Function calling: `get_weather(fecha)` validado por Pydantic y con registros en `logs/weather.log`.

## Resultados
- Prueba local: extracción e indexación realizadas correctamente sobre `data/TENERIFE.pdf` (8 chunks en mi ejecución de prueba). FAISS se construyó correctamente en el entorno de pruebas.
- `get_weather` devuelve respuestas deterministas y se registra en `logs/weather.log` (ya contiene 3 entradas de ejemplo tras la prueba).

## Instrucciones para el corrector
1. Activar entorno e instalar dependencias (ver `README.md`).
2. (Opcional) exportar `OPENAI_API_KEY` para usar embeddings reales.
3. Abrir y ejecutar `notebook.ipynb` desde el inicio.
4. Revisar `logs/weather.log` para ver invocaciones a la función.

## Limitaciones y mejoras futuras
- Añadir serialización/persistencia del índice en `artifacts/` y cargarlo en el notebook.
- Implementar llamadas reales al LLM con gestión superior de prompts y citations.
- Mejorar la memoria con embeddings de conversaciones y políticas de resumen.

## Conclusión
El repositorio contiene un prototipo funcional que cubre los requisitos esenciales; con la adición de las celdas finales en el notebook (persistencia del índice, métricas y ejemplos de prompts), estará listo para entregarse y optar a la máxima nota.
