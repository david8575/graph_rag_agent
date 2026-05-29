from contextlib import asynccontextmanager
from fastapi import FastAPI
from api.routes import pipeline, qa, status
from api.scheduler import start_scheduler, stop_scheduler

@asynccontextmanager
async def lifespan(app: FastAPI):
    start_scheduler()
    yield
    stop_scheduler()

app = FastAPI(title="Graph RAG Agent", lifespan=lifespan)

app.include_router(pipeline.router, prefix="/pipeline", tags=["Pipeline"])
app.include_router(qa.router, tags=["QA"])
app.include_router(status.router, tags=["Status"])
