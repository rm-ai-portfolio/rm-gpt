import os
import tempfile
import logging
from typing import Dict, List, Optional
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import settings

logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self._vector_store_cache: Dict[str, Optional[FAISS]] = {}

    def _get_loader(self, tmp_path: str, file_ext: str):
        file_ext = file_ext.lower()
        if file_ext == ".pdf":
            return PyPDFLoader(tmp_path)
        elif file_ext == ".docx":
            return Docx2txtLoader(tmp_path)
        elif file_ext == ".doc":
            try:
                from langchain_community.document_loaders import UnstructuredWordDocumentLoader
                return UnstructuredWordDocumentLoader(tmp_path)
            except ImportError:
                raise ImportError(
                    ".doc support requires 'unstructured' and 'python-docx'. "
                    "Install: pip install unstructured python-docx"
                )
        raise ValueError(f"Unsupported file type: {file_ext}")

    def process_files(self, uploaded_files: list) -> FAISS:
        all_chunks = []
        for uploaded_file in uploaded_files:
            file_ext = os.path.splitext(uploaded_file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            try:
                loader = self._get_loader(tmp_path, file_ext)
                docs = loader.load()
                chunks = self.text_splitter.split_documents(docs)
                all_chunks.extend(chunks)
            finally:
                os.unlink(tmp_path)

        if not all_chunks:
            raise ValueError("No valid documents processed.")
            
        return FAISS.from_documents(all_chunks, self.embeddings)

    def save_vector_store(self, vector_store: FAISS, session_id: str) -> None:
        path = os.path.join(settings.faiss_dir, session_id)
        os.makedirs(path, exist_ok=True)
        vector_store.save_local(path)
        self._vector_store_cache[session_id] = vector_store
        logger.info(f"Saved FAISS index to {path}")

    def _load_vector_store(self, session_id: str) -> Optional[FAISS]:
        path = os.path.join(settings.faiss_dir, session_id)
        if not os.path.exists(path):
            return None
        return FAISS.load_local(
            path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )

    def get_vector_store(self, session_id: str) -> Optional[FAISS]:
        if session_id not in self._vector_store_cache:
            self._vector_store_cache[session_id] = self._load_vector_store(session_id)
        return self._vector_store_cache.get(session_id)

    @staticmethod
    def format_docs(docs) -> str:
        return "\n\n".join(doc.page_content for doc in docs)
