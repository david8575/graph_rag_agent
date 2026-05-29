import json
import os
from langchain_ollama import OllamaLLM

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUMMARIZED_PATH = os.path.join(BASE_DIR, "data", "summarized.json")
OLLAMA_URL = "http://localhost:11434"
CHAT_MODEL = "gemma4:e4b"

llm = OllamaLLM(model=CHAT_MODEL, base_url=OLLAMA_URL)

def summarize_post(post: dict) -> dict:
    prompt = f"""
    다음 기술 뉴스 게시물 제목을 보고 2~3문장으로 한국어 요약을 작성해줘
    어떤 기술이고, 왜 주목할만 한지를 중심으로 작성해줘
    마크 다운 기호 등을 사용하지 말고 순수 텍스트로만 작성해줘

    제목:{post["title"]}

    요약:
    """

    summary = llm.invoke(prompt).strip()

    return {
        **post,
        "summary": summary,
        "summarized_post":""
    }

def summarize_posts(posts: list) -> list:
    summarized = []
    for i, post in enumerate(posts):
        print(f"  [{i+1}/{len(posts)}] summarizing: {post['title'][:30]}...")
        try:
            result = summarize_post(post)
            summarized.append(result)
        except Exception as e:
            print(f"  요약 실패: {e}")
            summarized.append({**post, "summary": "failed"})
    return summarized

def save_summarized(new_summarized: list) -> None:
    if os.path.exists(SUMMARIZED_PATH):
        with open(SUMMARIZED_PATH, "r", encoding="utf-8") as f:
            existing = json.load(f)
    else:
        existing = []

    existing_map = {item["url"]: item for item in existing}

    for item in new_summarized:
        existing_map[item["url"]] = item 

    merged = list(existing_map.values())

    os.makedirs(os.path.dirname(SUMMARIZED_PATH), exist_ok=True)
    with open(SUMMARIZED_PATH, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"[saved]: {SUMMARIZED_PATH} (totally {len(merged)} summaries)")