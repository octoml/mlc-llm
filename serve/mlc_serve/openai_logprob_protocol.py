from typing import List, Optional

from pydantic import BaseModel

class TopLogprobs(BaseModel):
    """An OpenAI API compatible schema for logprobs output."""

    token: str
    logprob: float
    bytes: Optional[List] = None


class LogprobsContent(BaseModel):
    """An OpenAI API compatible schema for logprobs output."""

    token: str
    logprob: float
    bytes: Optional[List] = None
    top_logprobs: List[TopLogprobs]  # It can be empty


class Logprobs(BaseModel):
    """
    An OpenAI API compatible schema for logprobs output.
    See details in https://platform.openai.com/docs/api-reference/chat/object#chat-create-logprobs
    """

    content: List[LogprobsContent]
