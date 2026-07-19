from typing import List, Literal
from pydantic import BaseModel

class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str

class ChatSession(BaseModel):
    id: str
    title: str
    messages: List[Message] = []
    file_names: List[str] = []
    has_vector_store: bool = False
