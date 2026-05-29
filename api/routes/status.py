import json
import os
from fastapi import APIRouter
from fastapi.responses import FileResponse

router = APIRouter()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@router.get("/status")
async def status():
    result = {}

    posts_path = os.path.join(BASE_DIR, "data", "posts.json")
    summarized_path = os.path.join(BASE_DIR, "data", "summarized.json")
    graph_path = os.path.join(BASE_DIR, "data", "graph.json")

    result["articles"] = len(json.load(open(posts_path, encoding="utf-8"))) if os.path.exists(posts_path) else 0
    result["summarized"] = len(json.load(open(summarized_path, encoding="utf-8"))) if os.path.exists(summarized_path) else 0

    if os.path.exists(graph_path):
        g = json.load(open(graph_path, encoding="utf-8"))
        result["nodes"] = len(g.get("nodes", []))
        result["edges"] = len(g.get("links", []))
    else:
        result["nodes"] = 0
        result["edges"] = 0

    return result

@router.get("/graph")
async def graph():
    return FileResponse(os.path.join(BASE_DIR, "data", "graph.html"), media_type="text/html")

@router.get("/graph/data")
async def graph_data():
    graph_path = os.path.join(BASE_DIR, "data", "graph.json")

    if not os.path.exists(graph_path):
        return {
            "nodes": [],
            "links": []
        }

    with open(graph_path, encoding="utf-8") as f:
        return json.load(f)