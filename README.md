# Chat With Your Documents - RAG Application

A lightweight Retrieval-Augmented Generation (RAG) application built with Streamlit and Pure Python.

## Features
- **PDF Understanding:** Extracts text from any PDF document.
- **Document Chunking:** Splits long text into overlapping chunks to preserve context.
- **Pure Python Embeddings:** Implements TF-IDF from scratch without heavy external dependencies. This avoids DLL and compatibility issues on newer Python versions.
- **Vector Search:** Uses Cosine Similarity to find the most relevant chunks for a user query.
- **RAG Pipeline:** Retrieves context and generates accurate, document-grounded answers.

## Tech Stack
- Python
- Streamlit
- PyPDF
- Pure Python (No sklearn / sentence-transformers)

## How to Run Locally
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
