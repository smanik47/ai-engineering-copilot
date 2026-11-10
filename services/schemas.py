
from pydantic import BaseModel, Field
from typing import Optional


class ChatRequest(BaseModel):

    question: str = Field(
        min_length=1,
        max_length=4000
    )

    conversation_id: Optional[str] = None

    top_k: int = Field(
        default=3,
        ge=1,
        le=10
    )


class SourceResponse(BaseModel):

    chunk_id: str
    source: str
    page: int
    similarity_score: float


class ChatResponse(BaseModel):

    conversation_id: str
    question: str
    answer: str
    sources: list[SourceResponse]
