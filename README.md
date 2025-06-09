#  RAG-Based Semantic Quote Retrieval System

This project is a **Retrieval-Augmented Generation (RAG)** pipeline that enables users to retrieve meaningful quotes based on semantic queries. The system uses a **fine-tuned BGE encoder model for embedding**, **FAISS for dense retrieval**, and a **FLAN-T5 model** for structured answer generation — all deployed through a user-friendly **Streamlit web app**.

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

