import asyncio
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

class QuestionRequest(BaseModel):
    question: str

@router.post("/qa")
async def ask_question(request: QuestionRequest):
    from agents.qa_agent import ask
    answer = await asyncio.to_thread(ask, request.question)

    return {
        "question": request.question,
        "answer": answer
    }