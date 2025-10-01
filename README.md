# PDF-RAG-QA


## Overview
PDF-RAG-QA is a local Python-based tool for **multimodal Retrieval-Augmented Generation (RAG)** over PDFs.  
It allows users to ask natural language questions about complex PDF documents containing text, tables, and images, using **Google Gemini LLM** and **FAISS** for accurate context retrieval.

## Features
- Extracts text, tables, and images from PDFs using **PyMuPDF**.
- Converts tables into pandas DataFrames for structured retrieval.
- Uses an **LLM-powered query expansion module** to improve answer accuracy.
- Interactive **Gradio interface** for PDF upload and question answering.

## Tech Stack
- Python  
- PyMuPDF  
- FAISS  
- LangChain  
- Gradio  
- Google Gemini LLM  

## Installation
1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd <repo-folder>
