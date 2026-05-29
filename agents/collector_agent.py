import asyncio
import json
import os
from datetime import datetime, timezone
from langchain_mcp_adapters.client import MultiServerMCPClient

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVE_PATH = os.path.join(BASE_DIR, 'data', 'posts.json')
GEEKNEWS_COMMAND = "python"
GEEKNEWS_ARGS = os.path.join(BASE_DIR, "GeekNews-MCP-Server", "main.py")

async def fetch_articles():
    client = MultiServerMCPClient(
        {
            "geeknews": {
                "command": GEEKNEWS_COMMAND,
                "args": [GEEKNEWS_ARGS],
                "transport": "stdio"
            }
        }
    )

    tools = await client.get_tools()
    get_article_tool = None

    for tool in tools:
        if tool.name == "get_articles":
            get_article_tool = tool
            break

    if get_article_tool is None:
        raise RuntimeError("get_article tool not found")
    
    top = await get_article_tool.ainvoke({"type": "top", "limit": 30})
    new = await get_article_tool.ainvoke({"type": "new", "limit": 30})
    collected_at = datetime.now(timezone.utc).isoformat()

    seen_urls = set()
    parsed_posts = []
    for item in top + new:
        data = json.loads(item["text"])
        if data["url"] in seen_urls:
            continue
        seen_urls.add(data["url"])
        data.pop("rank", None)
        data["collected_at"] = collected_at
        parsed_posts.append(data)

    return parsed_posts

def save_posts(new_posts: list):
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    existing = []
    if os.path.exists(SAVE_PATH):
        with open(SAVE_PATH, "r", encoding="utf-8") as f:
            existing = json.load(f)

    existing_urls = {p["url"] for p in existing}
    merged = existing + [p for p in new_posts if p["url"] not in existing_urls]

    with open(SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"[saved] total: {len(merged)} (+{len(merged) - len(existing)} new)")

def detect_new_posts(new_posts: list, saved_path: str) -> list:
    if not os.path.exists(saved_path):
        return new_posts
    
    with open(saved_path, "r", encoding="utf-8") as f:
        old_posts = json.load(f)

    old_urls = {post["url"] for post in old_posts}
    new_only = [post for post in new_posts if post["url"] not in old_urls]

    return new_only

async def main():
    print(f"[gathering posts]")
    posts = await fetch_articles()
    print(f"[completed]: {len(posts)} posts")

    new_posts = detect_new_posts(posts, SAVE_PATH)
    print(f"[new posts] : {len(new_posts)}")

    save_posts(posts)
