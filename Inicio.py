import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re
from nltk.stem import SnowballStemmer

# Configuración de la página
st.set_page_config(page_title="Hotel Transilvania - Demo TF-IDF", layout="wide")

# Título
st.title("🧛 Demo TF-IDF en el Hotel Transilvania")

# Imagen grande debajo del título
st.image("castle.jpg", use_container_width=True)

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
    question = st.text_input("❓ Escribe tu pregunta:", "¿Quién está en el cuarto 305?")

with col2:
    st.markdown("### 💡 Preguntas sugeridas:")
    
    # Preguntas sobre los cuartos
    if st.button("🔢 ¿Quién está en el cuarto 101?", use_container_width=True):
        st.session_state.question = "¿Quién está en el cuarto 101?"
        st.rerun()
    
    if st.button("🔢 ¿Quién está en el cuarto 202?", use_container_width=True):
        st.session_state.question = "¿Quién está en el cuarto 202?"
        st.rerun()
        
    if st.button("🔢 ¿Quién está en el cuarto 305?", use_container_width=True):
        st.session_state.question = "¿Quién está en el cuarto 305?"
        st.rerun()
        
    if st.button("🔢 ¿Quién está en el cuarto 408?", use_container_width=True):
        st.session_state.question = "¿Quién está en el cuarto 408?"
        st.rerun()
        
    if st.button("🔢 ¿Quién está en el cuarto 512?", use_container_width=True):
        st.session_state.question = "¿Quién está en el cuarto 512?"
        st.rerun()
    
    if st.button("🔢 ¿Quién está en el cuarto 618?", use_container_width=True):
        st.session_state.question = "¿Quién está en el cuarto 618?"
        st.rerun()
    
    if st.button("🔢 ¿Quién está en el cuarto 723?", use_container_width=True):
        st.session_state.question = "¿Quién está en el cuarto 723?"
        st.rerun()
    
    if st.button("🔢 ¿Quién está en el cuarto 831?", use_container_width=True):
        st.session_state.question = "¿Quién está en el cuarto 831?"
        st.rerun()
    
    if st.button("🔢 ¿Quién está en el cuarto 945?", use_container_width=True):
        st.session_state.question = "¿Quién está en el cuarto 945?"
        st.rerun()

# Actualizar pregunta si se seleccionó una sugerida
if 'question' in st.session_state:
    question = st.session_state.question

if st.button("🔍 Buscar en el hotel", type="primary"):
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
        st.markdown("### 🎯 Habitación encontrada")
        st.markdown(f"**Tu pregunta:** {question}")
        
        # Extraer el número de cuarto de la pregunta
        cuarto_pregunta = re.search(r'cuarto (\d+)', question)
        if cuarto_pregunta:
            cuarto_buscado = cuarto_pregunta.group(1)
            
            # Buscar si el cuarto existe en los documentos
            cuarto_encontrado = False
            for doc in documents:
                if f"cuarto {cuarto_buscado}" in doc.lower():
                    cuarto_encontrado = True
                    st.success(f"**Respuesta:** {doc}")
                    st.info(f"📈 Similitud: {best_score:.3f}")
                    
                    # Mostrar el resultado de forma destacada
                    st.markdown(f"""
                    <div style='background-color: #2e2e2e; padding: 20px; border-radius: 10px; border: 2px solid #ff6b6b;'>
                        <h2 style='color: #ff6b6b; text-align: center;'>🏨 ¡HABITACIÓN ENCONTRADA!</h2>
                        <h3 style='color: white; text-align: center;'>En el cuarto {cuarto_buscado} está: {doc.split('está')[0].strip() if 'está' in doc else doc.split('en')[0].strip()} 🎉</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    st.balloons()
                    break
            
            if not cuarto_encontrado:
                # Si no se encuentra el cuarto exacto, mostrar la mejor coincidencia
                st.warning(f"No se encontró el cuarto {cuarto_buscado} exactamente, pero la mejor coincidencia es:")
                st.success(f"**{best_doc}**")
                st.info(f"📈 Similitud: {best_score:.3f}")
        else:
            # Si la pregunta no tiene número de cuarto, mostrar la mejor coincidencia
            st.success(f"**Respuesta:** {best_doc}")
            st.info(f"📈 Similitud: {best_score:.3f}")

st.markdown("---")
st.caption("🧛 Hotel Transilvania - Encuentra quién está en cada habitación usando TF-IDF")
