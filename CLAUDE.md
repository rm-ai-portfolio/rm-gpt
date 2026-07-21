# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

This is a single-user Streamlit chat application ("RM GPT") with optional RAG over uploaded PDF and Word documents. The app uses Ollama for the LLM and HuggingFace Inference API for embeddings, with FAISS for local vector storage and JSON-backed chat sessions.

## Common commands

There is no build, lint, or test setup in this repo. Development is driven by Streamlit and pip.

- Install dependencies: `pip install -r requirements.txt`
- Run the app: `streamlit run src/app.py`
- Run the app on a specific port: `streamlit run src/app.py --server.port 8501`

The project uses a local virtual environment at `.venv`.

## Configuration

Runtime settings are loaded from a `.env` file via `src/config.py` using `pydantic-settings`. Copy `.env.example` to `.env` and fill in real values.

Required environment variables:
- `OLLAMA_API_KEY` — API key sent as a Bearer token to the Ollama endpoint.
- `LLM_MODEL` — Ollama model name (example in `.env.example`: `glm-5.2:cloud`).
- `OLLAMA_BASE_URL` — Base URL for the Ollama-compatible API.
- `HF_TOKEN` — HuggingFace access token for the inference API and embeddings.
- `HF_EMBEDDING_MODEL` — HuggingFace embedding model (default: `google/embeddinggemma-300m`).

The `.env.example` also contains `LANGSMITH_*` variables, but the application code does not currently wire up LangSmith tracing.

## Architecture

`src/app.py` is the Streamlit entry point. It renders the sidebar chat list, the active session's messages, a file uploader, and the chat input. It initializes three long-lived service objects in `st.session_state` and orchestrates them.

`src/config.py` defines `Settings` and `settings`, which reads the `.env` file and ensures the `data_dir` and `faiss_dir` directories exist. Most modules import `settings` from this singleton.

`src/models.py` holds the Pydantic models: `Message` with `role` and `content`, and `ChatSession` with id, title, messages, attached file names, and a `has_vector_store` flag.

`src/services/llm_service.py` wires the LLM and embeddings:
- Uses `ChatOllama` from `langchain_ollama` with `client_kwargs` to inject an `Authorization` Bearer header from `OLLAMA_API_KEY`.
- Uses a custom `HFInferenceEmbeddings` class for embedding documents and queries through the HuggingFace Inference API.
- Provides two LCEL chains: `standard_chain` for plain Q&A and `rag_chain` when context is supplied.

`src/services/embeddings_service.py` implements `HFInferenceEmbeddings`, a custom `langchain_core.embeddings.Embeddings` adapter around `huggingface_hub.InferenceClient.feature_extraction`. It normalizes the returned tensor shape so that scalar, 2D, and higher-dimensional outputs are flattened into a single vector.

`src/services/rag_service.py` handles document ingestion and retrieval:
- Accepts Streamlit `UploadedFile` objects, writes them to temporary files, and loads them with `langchain_unstructured.UnstructuredLoader` using a `by_title` chunking strategy.
- Builds a `FAISS` index from the chunks with the embeddings model.
- Persists each session's vector store under `data/faiss_indexes/<session_id>/`.
- Caches loaded indexes in memory (`_vector_store_cache`) and reloads from disk on demand with `allow_dangerous_deserialization=True`.

`src/services/session_manager.py` persists chat sessions to `data/sessions.json` using `filelock` for basic concurrency safety. Sessions are stored as a JSON dict keyed by session id. It auto-creates a first session if none exists, and loads the most recently active session on startup.

## State and data flow

1. On startup, `SessionManager` loads `data/sessions.json` and creates/selects an active session.
2. The user selects or creates a chat session from the sidebar.
3. File uploads flow through `RAGService.process_files`, which creates a FAISS index, saves it to disk, and records the file names in the active session via `SessionManager.update_session_files`.
4. When a message is sent, `app.py` checks `active_session.has_vector_store`:
   - If true, it loads the session's FAISS index, retrieves relevant chunks, and uses `LLMService.generate_response(input, context)`.
   - If false, it calls `LLMService.generate_response(input)` with no context.
5. `SessionManager.add_message` appends the user and assistant messages to the active session and writes the updated sessions file.

## Files and directories

- `data/sessions.json` — persisted chat sessions and messages. Ignored by git.
- `data/faiss_indexes/<session_id>/` — per-session FAISS vector stores. Ignored by git.
- `.env` — secrets and runtime config. Ignored by git.
- `requirements.txt` — direct dependencies. There is no lock file.
