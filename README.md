# [PDF Knowledge Bot](https://pdf-demo-eight.vercel.app/)

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![LlamaIndex](https://img.shields.io/badge/LlamaIndex-8B5CF6?style=for-the-badge&logoColor=white)](https://www.llamaindex.ai/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=111111)](https://huggingface.co/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![pypdf](https://img.shields.io/badge/pypdf-D62828?style=for-the-badge&logoColor=white)](https://pypdf.readthedocs.io/)
[![Vercel](https://img.shields.io/badge/Vercel-171717?style=for-the-badge&logo=vercel&logoColor=white)](https://vercel.com/)

Upload a text-based PDF, ask a question, and inspect relevant passages with clickable page citations.

## How it works

The deployed web app extracts page text with pypdf, creates overlapping chunks, and ranks them using BM25. It returns an explicit no-match result when no query terms are found. Optional generated answers use the retrieved excerpts and require a configured model provider.

Files are processed per request. The application does not persist uploaded documents.

## Implementations

- `web/`: deployed passage search and optional answer generation.
- `app.py`: separate Streamlit/LlamaIndex implementation with local embeddings.

[Provider configuration](docs/generation.md) explains how to enable generation. The provider-backed flow still needs live verification after credentials are configured.

## Run the Streamlit implementation

```bash
pip install -r requirements.txt
streamlit run app.py
```
