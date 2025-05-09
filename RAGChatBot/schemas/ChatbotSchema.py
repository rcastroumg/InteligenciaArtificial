from pydantic import BaseModel
from typing import Optional

class DocumentInput(BaseModel):
    id: str
    text: str
    metadata: Optional[dict] = {}

class QueryInput(BaseModel):
    conversation_id: str
    query: str
    top_k: int = 3

class NewConversationInput(BaseModel):
    user: str