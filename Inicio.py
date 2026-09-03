import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re
from nltk.stem import SnowballStemmer

# Configuración de la página
st.set_page_config(page_title="Hotel Transilvania - Demo TF-IDF", layout="wide")

# Título con imagen
col_title, col_img = st.columns([4, 1])
with col_title:
    st.title("🧛 Demo TF-IDF en el Hotel Transilvania")
with col_img:
    st.image("castle.jpg", width=100)

st.markdown("---")

# Documentos de ejemplo - Monstruos en el Hotel Transilvania
default_docs = """El conde Drácula está en el cuarto 101.
La momia descansa en el cuarto 202.
El hombre lobo aúlla en el cuarto 305.
Frankestein camina en el cuarto 408.
La novia de Frankestein se peina en el cuarto 409.
El fantasma flota en el cuarto 512.
La bruja prepara pociones en el cuarto 618.
El esqueleto baila en el cuarto 723.
El yeti vive en el cuarto 831.
El chupacabras duerme en el cuarto 945."""

# Stemmer en español
stemmer = SnowballStemmer("spanish")

def tokenize_and_stem(text):
    # Minúsculas
    text = text.lower()
    # Solo letras españolas y espacios
    text = re.sub(r'[^a-záéíóúüñ\s]', ' ', text)
    # Tokenizar
    tokens = [t for t in text.split() if len(t) > 1]
    # Aplicar stemming
    stems = [stemmer.stem(t) for t in tokens]
    return stems

# Layout en dos columnas
col1, col2 = st.columns([2, 1])

with col1:
    text_input = st.text_area("📝 Habitaciones del hotel (uno por línea):", default_docs, height=200)
    question = st.text_input("❓ Escribe tu pregunta:", "¿Dónde está el hombre lobo?")

with col2:
    st.markdown("### 💡 Preguntas sugeridas:")
    
    # Preguntas sobre los monstruos
    if st.button("🦁 ¿Dónde está el hombre lobo?", use_container_width=True):
        st.session_state.question = "¿Dónde está el hombre lobo?"
        st.rerun()
    
    if st.button("🧛 ¿Dónde está el conde Drácula?", use_container_width=True):
        st.session_state.question = "¿Dónde está el conde Drácula?"
        st.rerun()
        
    if st.button("🧟 ¿Dónde está Frankestein?", use_container_width=True):
        st.session_state.question = "¿Dónde está Frankestein?"
        st.rerun()
        
    if st.button("🧙 ¿Dónde está la bruja?", use_container_width=True):
        st.session_state.question = "¿Dónde está la bruja?"
        st.rerun()
        
    if st.button("👻 ¿Dónde está el fantasma?", use_container_width=True):
        st.session_state.question = "¿Dónde está el fantasma?"
        st.rerun()
    
    if st.button("💀 ¿Dónde está el esqueleto?", use_container_width=True):
        st.session_state.question = "¿Dónde está el esqueleto?"
        st.rerun()
    
    if st.button("🐺 ¿Dónde está el yeti?", use_container_width=True):
        st.session_state.question = "¿Dónde está el yeti?"
        st.rerun()

# Actualizar pregunta si se seleccionó una sugerida
if 'question' in st.session_state:
    question = st.session_state.question

if st.button("🔍 Buscar monstruo", type="primary"):
    documents = [d.strip() for d in text_input.split("\n") if d.strip()]
    
    if len(documents) < 1:
        st.error("⚠️ Ingresa al menos una habitación.")
    elif not question.strip():
        st.error("⚠️ Escribe una pregunta.")
    else:
        # Crear vectorizador TF-IDF
        vectorizer = TfidfVectorizer(
            tokenizer=tokenize_and_stem,
            min_df=1  # Incluir todas las palabras
        )
        
        # Ajustar con documentos
        X = vectorizer.fit_transform(documents)
        
        # Mostrar matriz TF-IDF
        st.markdown("### 📊 Matriz TF-IDF del Hotel")
        df_tfidf = pd.DataFrame(
            X.toarray(),
            columns=vectorizer.get_feature_names_out(),
            index=[f"Habitación {i+1}" for i in range(len(documents))]
        )
        st.dataframe(df_tfidf.round(3), use_container_width=True)
        
        # Calcular similitud con la pregunta
        question_vec = vectorizer.transform([question])
        similarities = cosine_similarity(question_vec, X).flatten()
        
        # Encontrar mejor respuesta
        best_idx = similarities.argmax()
        best_doc = documents[best_idx]
        best_score = similarities[best_idx]
        
        # Mostrar respuesta
        st.markdown("### 🎯 Monstruo encontrado")
        st.markdown(f"**Tu pregunta:** {question}")
        
        # Extraer el número de cuarto de la respuesta
        cuarto_match = re.search(r'cuarto (\d+)', best_doc)
        if cuarto_match:
            cuarto = cuarto_match.group(1)
            st.success(f"**Respuesta:** {best_doc}")
            st.info(f"📈 Similitud: {best_score:.3f}")
            
            # Mostrar un mensaje más amigable
            monstruo = best_doc.split('está')[0].strip() if 'está' in best_doc else best_doc.split('en')[0].strip()
            st.balloons()
            st.markdown(f"### 🏨 El monstruo está en el cuarto {cuarto}!")
        else:
            st.success(f"**Respuesta:** {best_doc}")
            st.info(f"📈 Similitud: {best_score:.3f}")

st.markdown("---")
st.caption("🧛 Hotel Transilvania - Encuentra a tus monstruos favoritos usando TF-IDF")
