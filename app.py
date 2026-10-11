import streamlit as st
from pypdf import PdfReader
import re
import math
from collections import Counter

st.set_page_config(page_title="RAG Chat - Module 4")
st.title("Chat With Your Documents - Module 4")

def split_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return [c for c in chunks if c.strip()]

def tokenize(text):
    return re.findall(r'\w+', text.lower())

def compute_tfidf_vectors(chunks):
    tfs = [Counter(tokenize(c)) for c in chunks]
    df = Counter()
    for tf in tfs:
        for word in tf.keys():
            df[word] += 1
    N = len(chunks)
    idf = {w: math.log(N / (1 + freq)) for w, freq in df.items()}
    vectors = []
    for tf in tfs:
        vec = {w: count * idf[w] for w, count in tf.items()}
        vectors.append(vec)
    return vectors, idf

def cosine_sim(vec1, vec2):
    common = set(vec1.keys()) & set(vec2.keys())
    dot = sum(vec1[w] * vec2[w] for w in common)
    mag1 = math.sqrt(sum(v*v for v in vec1.values()))
    mag2 = math.sqrt(sum(v*v for v in vec2.values()))
    if mag1 == 0 or mag2 == 0:
        return 0
    return dot / (mag1 * mag2)

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""

    chunks = split_text(text)
    st.success(f"Chunking completed: {len(chunks)} chunks created.")

    vectors, idf = compute_tfidf_vectors(chunks)
    st.success("Embeddings completed using Pure Python TF-IDF.")

    question = st.text_input("Ask a question from the document:")
    if question:
        q_tf = Counter(tokenize(question))
        q_vec = {w: q_tf[w] * idf.get(w, 0) for w in q_tf}
        scores = [cosine_sim(q_vec, v) for v in vectors]
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:3]

        st.subheader("Retrieved Context (Vector Search):")
        for i in top_idx:
            st.write(f"- {chunks[i][:400]}...")

        st.subheader("Answer (RAG):")
        st.write(chunks[top_idx[0]])