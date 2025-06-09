#  RAG-Based Semantic Quote Retrieval System

This project is a **Retrieval-Augmented Generation (RAG)** pipeline that allows users to retrieve and generate quotes based on semantic queries. It combines **fine-tuned sentence embeddings**, **FAISS vector search**, and **LLM-based generation** via OpenAI to build an intelligent quote search system.

---

##  Features

-  **Semantic search**: Find relevant quotes using natural language.
-  **BGE-based fine-tuned model**: Trained on the quotes dataset for query-quote matching.
-  **Fast FAISS vector search**: For efficient nearest neighbor retrieval.
-  **LLM generation**: Generates structured responses based on retrieved quotes.
-  **Streamlit Web App**: Interactive UI for users.
-  **RAGAS Evaluation**: Evaluates the pipeline using metrics like ROUGE, BLEU, METEOR, faithfulness, and answer relevancy.

---

##  Project Structure
├── app.py # Streamlit app
├── train_model.py # Training script for BGE fine-tuning
├── requirements.txt # Dependencies for Streamlit deployment
├── quote_metadata.csv # CSV file with quote-author-tags
├── quote_index.faiss # FAISS index of quote embeddings
├── fine_tuned_bge_quote_model/ # Directory containing fine-tuned embedding model
└── README.md # Project documentation
##  Install Requirements
pip install -r requirements.txt
##  To run the streamlit app
streamlit run app.py

