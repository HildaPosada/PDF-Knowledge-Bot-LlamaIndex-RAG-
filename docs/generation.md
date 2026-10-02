# Document question answering

The deployed API chunks page text, ranks passages with BM25, and links citations back to the uploaded PDF. Retrieval works without a key. Files are not persisted by this application.

For optional generation, set `OPENAI_API_KEY` or `HUGGINGFACEHUB_API_TOKEN` and `LLM_MODEL` in Vercel. Never commit keys. The UI enables generation only when configured, and explicitly states that excerpts are sent to the provider. Generation has not been verified until a provider is configured and tested. BM25 is lexical retrieval, not semantic embeddings. The separate LlamaIndex app remains a different implementation.
