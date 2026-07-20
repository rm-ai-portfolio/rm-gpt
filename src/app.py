import streamlit as st
import logging
import os
from dotenv  import load_dotenv
from services.llm_service import LLMService
from services.rag_service import RAGService
from services.session_manager import SessionManager

logging.basicConfig(level=logging.INFO)

load_dotenv()

os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HF_TOKEN")

def init_state() -> None:
    if "llm_service" not in st.session_state:
        st.session_state.llm_service = LLMService()
    if "rag_service" not in st.session_state:
        st.session_state.rag_service = RAGService(st.session_state.llm_service.get_embeddings())
    if "session_manager" not in st.session_state:
        st.session_state.session_manager = SessionManager()

def main() -> None:
    st.title("RM GPT")
    init_state()
    
    manager = st.session_state.session_manager
    llm_service = st.session_state.llm_service
    rag_service = st.session_state.rag_service
    
    with st.sidebar:
        st.title("Chats")
        if st.button("➕ New Chat", use_container_width=True):
            manager.create_new_session()
            st.rerun()
        st.divider()
        
        for session_id, session in manager.sessions.items():
            if st.button(session.title, key=session_id, use_container_width=True):
                manager.set_active_session(session_id)
                st.rerun()
    
    active_session = manager.get_active_session()
    if not active_session:
        st.warning("No active session.")
        return
        
    for message in active_session.messages:
        with st.chat_message(message.role):
            st.markdown(message.content)
            
    if active_session.file_names:
        st.caption(f"📄 Attached File(s): **{', '.join(active_session.file_names)}**")
        
    uploaded_files = st.file_uploader(
        "Attach PDF or DOCX files",
        type=["pdf", "doc", "docx"],
        accept_multiple_files=True,
        key=f"uploader_{active_session.id}"
    )
    
    if uploaded_files:
        new_file_names = [f.name for f in uploaded_files]
        if set(new_file_names) != set(active_session.file_names):
            with st.spinner("Processing files and creating vector store..."):
                try:
                    vector_store = rag_service.process_files(uploaded_files)
                    rag_service.save_vector_store(vector_store, active_session.id)
                    manager.update_session_files(new_file_names)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error processing files: {e}")
                    
    if prompt := st.chat_input("Enter your question:"):
        manager.add_message("user", prompt)
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            with st.spinner("Generating response..."):
                if active_session.has_vector_store:
                    vector_store = rag_service.get_vector_store(active_session.id)
                    if vector_store:
                        retriever = vector_store.as_retriever()
                        retrieved_docs = retriever.invoke(prompt)
                        context = rag_service.format_docs(retrieved_docs)
                        response = llm_service.generate_response(prompt, context)
                        
                        st.markdown(response)
                        with st.expander("📄 Show Source Documents"):
                            for i, doc in enumerate(retrieved_docs):
                                st.markdown(f"**Chunk {i+1}**")
                                st.text(doc.page_content)
                                st.caption(f"Metadata: {doc.metadata}")
                    else:
                        st.error("Vector store not found.")
                        response = "Error: Could not load vector store."
                else:
                    response = llm_service.generate_response(prompt)
                    st.markdown(response)
                    
        manager.add_message("assistant", response)

if __name__ == "__main__":
    main()
