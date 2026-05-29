import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from api.routes import pipeline, qa, status
from api.scheduler import start_scheduler, stop_scheduler
from dotenv import load_dotenv

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    start_scheduler()
    yield
    stop_scheduler()

app = FastAPI(title="Graph RAG Agent", lifespan=lifespan)

app.include_router(pipeline.router, prefix="/pipeline", tags=["Pipeline"])
app.include_router(qa.router, tags=["QA"])
app.include_router(status.router, tags=["Status"])
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return FileResponse("static/index.html")