import asyncio
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

class QuestionRequest(BaseModel):
    question: str

@router.post("/qa")
async def ask_question(req: QuestionRequest):
    from agents.qa_agent import ask

    result = await asyncio.to_thread(ask, req.question)
    
    return {
        "question": req.question, 
        **result
    }