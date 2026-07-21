import logging
from typing import Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from config import settings
from services.embeddings_service import HFInferenceEmbeddings

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self):
        self.llm = ChatOllama(
            model=settings.llm_model,
            base_url=settings.ollama_base_url,
            client_kwargs={"headers": {"Authorization": f"Bearer {settings.ollama_api_key}"}}
        )
        self.embeddings = HFInferenceEmbeddings(
            model=settings.hf_embedding_model,
            api_key=settings.hf_token
        )
        self.parser = StrOutputParser()
        
        self.standard_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that can answer questions."),
            ("user", "Question: {input}"),
        ])
        
        self.rag_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Use the following context to answer the user's question.\n\nContext:\n{context}"),
            ("user", "Question: {input}"),
        ])
        
        self.standard_chain = self.standard_prompt | self.llm | self.parser
        self.rag_chain = self.rag_prompt | self.llm | self.parser

    def get_embeddings(self) -> HuggingFaceEndpointEmbeddings:
        return self.embeddings

    def generate_response(self, input_text: str, context: Optional[str] = None) -> str:
        if context:
            return self.rag_chain.invoke({"context": context, "input": input_text})
        return self.standard_chain.invoke({"input": input_text})
