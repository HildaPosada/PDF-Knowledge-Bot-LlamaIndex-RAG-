# Web document search

This deployment extracts PDF text with pypdf and ranks passages by question keywords. Results quote the uploaded file and cite its page. It does not call an LLM or use LlamaIndex; the separate Streamlit application implements that configuration.

Vercel: Framework Other, Root Directory `web`. Limits: 3 MB uploads, 100 pages, unencrypted text PDFs. Scans need OCR. The app does not persist uploaded documents. Lexical matching can miss paraphrases and does not establish that a passage answers a question.
