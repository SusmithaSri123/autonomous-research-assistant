# Autonomous Research Assistant

A small Streamlit app that fetches papers from arXiv, summarizes them using Hugging Face transformers, and extracts keywords.

## Features
- Query arXiv and fetch paper title/summary/authors
- Summarization via transformers pipeline
- Simple TF-IDF keyword extraction
- Streamlit UI for quick browsing

## Setup (local)
```bash
# create & activate venv (Linux/macOS)
python -m venv .venv
source .venv/bin/activate

# or Windows PowerShell
python -m venv .venv
.venv\Scripts\Activate.ps1

# install dependencies
pip install -r requirements.txt

# run the app
streamlit run research_assistant.py
