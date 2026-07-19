import tempfile
from turtle import up
from langchain_community.document_loaders.base_o365 import CHUNK_SIZE
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tools import retriever
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


from ollama import embeddings
import streamlit as st
import os
from dotenv import load_dotenv


load_dotenv()

# Setup Prompt Template
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that can answer questions."),
    ("user", "Question: {input}"),
])

# New RAG prompt with context
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the following context to answer the user's question.\n\nContext:\n{context}"),
    ("user", "Question: {input}"),
])

# Setup Output Parser
parser = StrOutputParser()

# Setup LLM and chain
llm = ChatOllama(
    model="glm-5.2:cloud",
    base_url="https://ollama.com",
    client_kwargs={
        "headers": {
            "Authorization": f"Bearer {os.getenv('OLLAMA_API_KEY')}"
        }
    }
)


#Setup Embeddings
embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://localhost:11434"
)

standard_chain = prompt_template | llm | parser

# --- Session Management ---
def create_new_session():
    """Creates a new chat session and sets it as active."""
    st.session_state.session_counter += 1
    session_id = f"chat_{st.session_state.session_counter}"
    st.session_state.chat_sessions[session_id] = {
        "title": f"New Chat {st.session_state.session_counter}",
        "messages": [],
        "vector_store": None,
        "file_names": []
    }
    st.session_state.active_session_id = session_id

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def process_uploaded_files(uploaded_files):
    all_chunks= []

    for uploaded_file in uploaded_files:
        file_extension = os.path.splitext(uploaded_file.name)[1]
        print(f"File Suffix: {file_extension}")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        if file_extension == ".pdf":
            loader = PyPDFLoader(tmp_file_path)
        elif file_extension == ".doc":
            loader = Docx2txtLoader(tmp_file_path)
        else:
            st.error("Unsupported file type. Please upload a PDF or DOCX.")
            os.unlink(tmp_file_path)
            return None
        
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        file_chunks = text_splitter.split_documents(docs)

        all_chunks.extend(file_chunks)
        
        os.unlink(tmp_file_path)

    if not all_chunks:
        return None

    vector_store = FAISS.from_documents(all_chunks, embeddings)
    return vector_store


# Initialize state
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {}
    st.session_state.session_counter = 0
    create_new_session() # Start with one active chat

# --- Streamlit UI ---
st.title("RM GPT")

# Sidebar for sessions
with st.sidebar:
    st.title("Chats")
    if st.button("➕ New Chat", use_container_width=True):
        create_new_session()
    
    st.divider()
    
    # Display session buttons
    for session_id, session_data in st.session_state.chat_sessions.items():
        is_active = session_id == st.session_state.active_session_id
        if st.button(session_data["title"], key=session_id, use_container_width=True):
            st.session_state.active_session_id = session_id
            st.rerun() # Rerun to update the main chat view immediately

# --- Main Chat View ---
# Get the currently active session
active_session = st.session_state.chat_sessions[st.session_state.active_session_id]

# Display chat history
for message in active_session["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if active_session["file_names"]:
    st.caption(f" Attached File: **{active_session['file_names']}**")

if uploaded_files := st.file_uploader(
    "Attach a PDF or Doc or Docx file", 
    type=["pdf", "doc", "docx"],
    accept_multiple_files=True,
    key=f"uploader_{st.session_state.active_session_id}"
):
    new_file_names=[f.name for f in uploaded_files]

    if set(new_file_names) != set(active_session["file_names"]):
        with st.spinner("Processing file and createing vectore store..."):
            if vector_store := process_uploaded_files(uploaded_files):
                active_session["vector_store"] = vector_store
                active_session["file_names"] = new_file_names

                st.rerun()

# Process User Input
if prompt := st.chat_input("Enter your question:"):
    # Append user message to state and UI
    active_session["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Generating response..."):
            if active_session["vector_store"]:
                # 1. Retrieve documents explicitly
                retriever = active_session["vector_store"].as_retriever()
                retrieved_docs = retriever.invoke(prompt)
                
                # 2. Format context and invoke LLM
                context = format_docs(retrieved_docs)
                response = (rag_prompt | llm | parser).invoke({"context": context, "input": prompt})
                
                # 3. Display main response
                st.markdown(response)
                
                # 4. Show source documents in an expander
                with st.expander("📄 Show Source Documents"):
                    for i, doc in enumerate(retrieved_docs):
                        st.markdown(f"**Chunk {i+1}**")
                        st.text(doc.page_content)
                        st.caption(f"Metadata: {doc.metadata}")

            else:
                response = standard_chain.invoke({"input": prompt})
                st.markdown(response)
    
    # Append assistant message to state
    active_session["messages"].append({"role": "assistant", "content": response})