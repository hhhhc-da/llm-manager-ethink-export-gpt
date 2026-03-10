from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class Model(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str

class Message(BaseModel):
    role: str
    content: str

class Choice(BaseModel):
    index: int
    message: Message
    finish_reason: Optional[str] = None
    logprobs: Optional[Dict] = None

class ModelListResponse(BaseModel):
    data: List[Model]
    object: str = "list"

class OpenAIMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[OpenAIMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 512

class ChatCompletionResponse(BaseModel):
    id: str = "chatcmpl-123456789"
    object: str = "chat.completion"
    created: int = 1773141081
    model: str = "gpt-4.1"
    choices: List[Choice]
    usage: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}