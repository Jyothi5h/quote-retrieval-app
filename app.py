import streamlit as st
import pandas as pd
import numpy as np
import os
import faiss
import json
from sentence_transformers import SentenceTransformer

# Load model and data
model_path = os.path.join(os.path.dirname(__file__), "fine_tuned_bge_quote_model")
model = SentenceTransformer(model_path, device='cpu')
df = pd.read_csv("quote_metadata.csv")
corpus = df['quote'].tolist()
index = faiss.read_index("quotes_index.faiss")

# UI
st.title("🧠 Semantic Quote Search (RAG)")
query = st.text_input("🔍 Enter your query:")

if query:
    query_embedding = model.encode([query])[0]
    D, I = index.search(np.array([query_embedding]), k=5)

    results = [corpus[i] for i in I[0]]
    st.subheader("📜 Retrieved Quotes:")
    for i, quote in enumerate(results):
        st.write(f"{i+1}. {quote}")

    # Optionally, show scores
    if st.checkbox("Show similarity scores"):
        for i in range(len(D[0])):
            st.write(f"Score {i+1}: {D[0][i]:.4f}")
