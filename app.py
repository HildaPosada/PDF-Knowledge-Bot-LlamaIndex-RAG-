import os
# Make sure HF token from secrets is visible as environment variable
if "HUGGINGFACEHUB_API_TOKEN" in st.secrets:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]

import tempfile
import streamlit as st

from llama_index.core import VectorStoreIndex, Settings, StorageContext, load_index_from_storage
from llama_index.core import SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Choose your LLM backend:
USE_HF_API = st.secrets.get("USE_HF_API", True)  # set False to use Ollama (offline)

# LLMs
if USE_HF_API:
    from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
    # Free models that work well:
    # - "mistralai/Mistral-7B-Instruct-v0.2"
    # - "google/gemma-2-9b-it" (rate limits vary)
    LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
    llm = HuggingFaceInferenceAPI(
        model_name=LLM_MODEL,
        temperature=0.1,
        max_tokens=512,
    )
else:
    from llama_index.llms.huggingface import HuggingFaceLLM
    # Local via Ollama. HuggingFaceLLM wrapper calls a local endpoint, but
    # for simplest offline use we will just hit Ollama’s HTTP endpoint with a tiny helper.
    # To keep it simple, we will use the HF local pipeline fallback with a small model.
    # If you want pure Ollama, see the comment block below.
    #
    # Small local pipeline option (RAM friendly). You can replace with a larger GGUF if you have VRAM:
    # "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    llm = HuggingFaceLLM(
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        tokenizer_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        generate_kwargs={"max_new_tokens": 512, "temperature": 0.1},
        device_map="auto",
    )

# Embeddings: completely free and fast
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)
Settings.llm = llm

st.set_page_config(page_title="PDF Knowledge Bot", page_icon="⚡", layout="wide")
st.title("⚡ PDF Knowledge Bot")

st.markdown(
    "Upload a PDF, we build a local vector index, and you can ask questions. "
    "Default uses free Hugging Face API. Toggle to offline by setting `USE_HF_API=False` or via secrets."
)

# Simple toggle in UI for convenience
use_api_ui = st.toggle("Use Hugging Face free API", value=bool(USE_HF_API))
USE_HF_API = use_api_ui  # reflect toggle in runtime, but we keep the same llm object for session simplicity

uploaded_files = st.file_uploader("Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)

persist_dir = "storage"
os.makedirs(persist_dir, exist_ok=True)

def build_or_load_index(file_bytes_list):
    """Builds a new index if PDFs are provided. Otherwise attempts to load from disk."""
    if file_bytes_list:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write PDFs to temp, then read
            for i, fobj in enumerate(file_bytes_list):
                with open(os.path.join(tmpdir, f"doc_{i}.pdf"), "wb") as out:
                    out.write(fobj.read())
            docs = SimpleDirectoryReader(tmpdir).load_data()

        index = VectorStoreIndex.from_documents(docs, show_progress=True)
        # persist
        index.storage_context.persist(persist_dir=persist_dir)
        return index
    else:
        # try to load existing
        if os.path.isdir(persist_dir) and len(os.listdir(persist_dir)) > 0:
            storage = StorageContext.from_defaults(persist_dir=persist_dir)
            return load_index_from_storage(storage)
        return None

index = build_or_load_index(uploaded_files)

if index is None:
    st.info("Upload a PDF to build the index. Once built, it will be cached in the `storage/` folder.")
else:
    query_engine = index.as_query_engine(similarity_top_k=4, response_mode="compact")
    question = st.text_input("Ask a question about your PDF")
    if st.button("Ask") or (question and st.session_state.get("auto_run", False)):
        with st.spinner("Thinking..."):
            try:
                response = query_engine.query(question)
                st.markdown("### Answer")
                st.write(response.response)

                # Show sources
                if hasattr(response, "source_nodes"):
                    st.markdown("### Sources")
                    for i, node in enumerate(response.source_nodes, start=1):
                        st.write(f"{i}. score={getattr(node, 'score', None)}")
                        st.code(node.node.get_content()[:1200] + ("..." if len(node.node.get_content()) > 1200 else ""))

            except Exception as e:
                st.error(f"Error: {e}")