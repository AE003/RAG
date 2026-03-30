# RAG – Retrieval Augmented Generation

This project implements a simple **Retrieval Augmented Generation (RAG)** system that allows a language model to answer questions using information retrieved from documents.

Instead of relying only on the model’s internal knowledge, the system searches through stored documents, retrieves the most relevant content, and uses it as context to generate better responses.

## Project Files

- **app.py** – main application logic  
- **index.py** – builds the document index and stores embeddings  
- **requirements.txt** – project dependencies  

## How it Works

1. Documents are loaded from the data folder.
2. Text is split into smaller chunks.
3. Embeddings are generated for each chunk.
4. Embeddings are stored in a **Chroma vector database**.
5. When a user asks a question, the system retrieves relevant document chunks.
6. The retrieved context is used by the language model to generate an answer.

## Run the Project

Install dependencies:

```bash
pip install -r requirements.txt
