import asyncio
import json
import os
from fastapi import APIRouter, BackgroundTasks

router = APIRouter()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_status = {
    "collect": "idle",
    "summarize": "idle",
    "build-graph": "idle",
    "email": "idle",
    "pipeline": "idle"
}

async def _run_collect():
    _status["collect"] = "running"

    try: 
        from agents.collector_agent import fetch_articles, detect_new_posts, save_posts, SAVE_PATH

        posts = await fetch_articles()
        new_posts = detect_new_posts(posts, SAVE_PATH)
        save_posts(posts)

        _status["collect"] = f"done (new: {len(new_posts)})"

        return new_posts
    
    except Exception as e: 
        _status["collect"] = f"error: {e}"

        return []
    
async def _run_summarize(new_posts=None):
    _status["summarize"] = "running"

    try: 
        from agents.summarizer_agent import summarize_posts, save_summarized

        if new_posts is None:
            posts_path = os.path.join(BASE_DIR, "data", "posts.json")
            summarized_path = os.path.join(BASE_DIR, "data", "summarized.json")

            with open(posts_path, encoding="utf-8") as f:
                all_posts = json.load(f)

            existing_urls = set()

            if os.path.exists(summarized_path):
                with open(summarized_path, encoding="utf-8") as f:
                    existing_urls = {p["url"] for p in json.load(f)}

            new_posts = [p for p in all_posts if p["url"] not in existing_urls]

        if new_posts:
            summarized = await asyncio.to_thread(summarize_posts, new_posts)
            await asyncio.to_thread(save_summarized, summarized)

            _status["summarize"] = f"done ({len(summarized)} summarized)"
        
        else:
            _status["summarize"] = "done (no new posts)"
            
    except Exception as e:
        _status["summarize"] = f"error: {e}"

async def _run_build_graph():
    _status["build-graph"] = "running"

    try:
        summarized_path = os.path.join(BASE_DIR, "data", "summarized.json")
        with open(summarized_path, encoding="utf-8") as f:
            posts = json.load(f)

        from agents.graph_builder_agent import build_graph, save_graph
        from agents.visualizer_agent import visualize

        G = await asyncio.to_thread(build_graph, posts)
        await asyncio.to_thread(save_graph, G)
        await asyncio.to_thread(visualize)

        _status["build-graph"] = f"done (nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()})"

    except Exception as e:
        _status["build-graph"] = f"error: {e}"

async def _run_email(new_posts: list = None):
    _status["email"] = "running"
    try:
        from agents.email_agent import send_briefing

        if new_posts is not None:
            new_urls = {p["url"] for p in new_posts}
            summarized_path = os.path.join(BASE_DIR, "data", "summarized.json")
            with open(summarized_path, encoding="utf-8") as f:
                all_summarized = json.load(f)
            posts_to_send = [p for p in all_summarized if p["url"] in new_urls]
        else:
            posts_to_send = None

        await asyncio.to_thread(send_briefing, posts_to_send)
        _status["email"] = "done"
    except Exception as e:
        _status["email"] = f"error: {e}"



async def _run_pipeline():
    _status["pipeline"] = "running"

    try: 
        new_posts = await _run_collect()
        await _run_summarize(new_posts)
       

        if new_posts:
            await _run_build_graph()
            await _run_email(new_posts)

        _status["pipeline"] = "done"

    except Exception as e:
        _status["pipeline"] = f"error: {e}"

@router.post("/collect")
async def collect(background_tasks: BackgroundTasks):
    if _status["collect"] == "running":
        return {
            "status": "already runnning"
        }
    
    background_tasks.add_task(_run_collect)

    return {
        "status": "started",
        "task": "collect"
    }

@router.post("/summarize")
async def summarize(background_tasks: BackgroundTasks):
    if _status["summarize"] == "running":
        return {
            "status": "already running"
        }
    background_tasks.add_task(_run_summarize)

    return {
        "status": "started", 
        "task": "summarize"
    }

@router.post("/build-graph")
async def build_graph_route(background_tasks: BackgroundTasks):
    if _status["build-graph"] == "running":
        return {
            "status": "already running"
        }
    
    background_tasks.add_task(_run_build_graph)

    return {
        "status": "started", 
        "task": "build-graph"
    }

@router.post("/email")
async def send_email(background_tasks: BackgroundTasks):
    if _status["email"] == "running":
        return {
            "status": "already running"
        }
    background_tasks.add_task(_run_email)
    return {
        "status": "started",            
        "task": "email"
    }

@router.post("/run")
async def run_pipeline(background_tasks: BackgroundTasks):
    if _status["pipeline"] == "running":
        return {
            "status": "already running"
        }
    
    background_tasks.add_task(_run_pipeline)

    return {
        "status": "started", 
        "task": "pipeline"
    }

@router.get("/tasks")
async def task_status():
    return _status
