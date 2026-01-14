import streamlit as st
from pathlib import Path
from src.rag_tools import get_embeddings, get_weather, respond_with_rag
import numpy as np
import json

st.set_page_config(page_title='Asistente Turístico RAG', layout='wide')
st.title('Asistente Turístico con RAG + Function Calling')

# LLM Configuration (mockified for demo)
st.sidebar.header('Configuración LLM')
st.sidebar.info('API Key cargada desde variable de entorno `OPENAI_API_KEY` (si existe)')
temperature = st.sidebar.slider('Temperatura', 0.0, 2.0, 0.7)
max_tokens = st.sidebar.slider('Max tokens', 100, 2000, 500)
top_p = st.sidebar.slider('Top-P', 0.0, 1.0, 0.9)
st.sidebar.write(f'Parámetros activos: T={temperature}, Max={max_tokens}, P={top_p}')
use_llm_sidebar = st.sidebar.checkbox('Usar LLM (si OPENAI_API_KEY)', value=False)

# Load precomputed index
ART = Path('artifacts') / 'tenerife_index'
if 'chunks' not in st.session_state:
    try:
        chunks_file = ART / 'chunks.json'
        if chunks_file.exists():
            with open(chunks_file, 'r', encoding='utf-8') as f:
                st.session_state.chunks = json.load(f)
            # Load vectors
            vecs_file = ART / 'vectors.npy'
            if vecs_file.exists():
                st.session_state.vectors = np.load(vecs_file)
                st.session_state.loaded = True
                st.success(f'✓ Índice precalculado cargado: {len(st.session_state.chunks)} chunks')
            else:
                st.warning('Vectors no encontrados en artifacts/')
        else:
            st.warning('Index no encontrado. Ejecuta el notebook primero.')
    except Exception as e:
        st.error(f'Error al cargar index: {e}')
        st.session_state.loaded = False

# Search interface
st.header('Búsqueda Semántica')
query = st.text_input('Pregunta sobre Tenerife:')

if query and st.session_state.get('loaded'):
    try:
        qv = get_embeddings([query])[0]
        vecs = st.session_state.vectors
        
        # Cosine similarity
        def cosine(a, b):
            a_norm = a / (np.linalg.norm(a) + 1e-12)
            b_norm = b / (np.linalg.norm(b) + 1e-12)
            return float(np.dot(a_norm, b_norm))
        
        sims = [cosine(qv, v) for v in vecs]
        top_idx = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:3]
        
        st.subheader('Resultados:')
        for rank, i in enumerate(top_idx, 1):
            col1, col2 = st.columns([1, 10])
            with col1:
                st.metric('Score', f'{sims[i]:.3f}')
            with col2:
                st.write(f'**Chunk {i}**')
                st.write(st.session_state.chunks[i][:500] + '...')
        # Optionally call LLM to generate an answer with citations
        if use_llm_sidebar:
            try:
                res = respond_with_rag(query, vectors=st.session_state.vectors, chunks=st.session_state.chunks, index=None, top_k=3, use_llm=True)
                st.subheader('Respuesta generada (LLM + citas)')
                st.write(res['answer'])
                st.markdown('**Fuentes:** ' + ', '.join(res['sources']))
            except Exception as e:
                st.error(f'Error al generar respuesta LLM: {e}')
    except Exception as e:
        st.error(f'Error en búsqueda: {e}')

# Function calling demo
st.header('Function Calling: get_weather(fecha)')
col1, col2, col3 = st.columns(3)
with col1:
    if st.button('Weather 2026-01-14'):
        st.write(get_weather('2026-01-14'))
with col2:
    if st.button('Weather 2026-01-15'):
        st.write(get_weather('2026-01-15'))
with col3:
    if st.button('Weather 2026-01-16'):
        st.write(get_weather('2026-01-16'))

st.info('Logs guardados en `logs/weather.log`')

