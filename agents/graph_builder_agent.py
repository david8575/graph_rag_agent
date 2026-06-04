import json
import os
import numpy as np
import networkx as nx
from itertools import combinations
from langchain_ollama import OllamaEmbeddings, OllamaLLM

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUMMARIZED_PATH = os.path.join(BASE_DIR, "data", "summarized.json")
GRAPH_PATH = os.path.join(BASE_DIR, "data", "graph.json")
OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "mxbai-embed-large"
CHAT_MODEL = "gemma4:e4b"
SIMILAR_THRESHOLD = 0.90
REFERENCE_MIN_OVERLAP = 1

llm = OllamaLLM(model=CHAT_MODEL, base_url=OLLAMA_URL)
embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_URL)

def get_embedding(text: str) -> list:
    return embeddings.embed_query(text)

def extract_topics(summary: str) -> list[str]:
    print("    [debug] extract_topics func started")

    prompt = f"""
    다음 기술 뉴스를 요약해서 핵심 기술 키워드를 3-5개 추출해주세오
    쉼표로 구분하여 키워드만 나열하고, 다른 텍스트는 포함하지 마세요
    
    요약: {summary}

    키워드:
    """

    raw = llm.invoke(prompt).strip()
    topics = [t.strip() for t in raw.split(",") if t.strip()]

    return topics[:5]

def cosine_similarity(vec1: list, vec2: list) -> float:
    v1, v2 = np.array(vec1), np.array(vec2)

    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

def build_graph(posts: list) -> nx.DiGraph:
    if os.path.exists(GRAPH_PATH):
        G = load_graph()
        existing_urls = {n for n, d in G.nodes(data=True) if d.get("type") == "Article"}
        existing_embs = {n: d["embedding"] for n, d in G.nodes(data=True) if d.get("type") == "Article" and "embedding" in d}
        existing_topics = {n: d.get("topics", []) for n, d in G.nodes(data=True) if d.get("type") == "Article"}
    else:
        G = nx.DiGraph()
        existing_urls = set()
        existing_embs = {}
        existing_topics = {}

    new_posts = [p for p in posts if p["url"] not in existing_urls]

    if not new_posts:
        print("[graph] no new posts")
        return G

    print(f"[graph] {len(new_posts)} new / {len(existing_urls)} existing")

    new_embs = {}
    new_topics = {}

    print("(1/4) [Node Generating]")
    for post in new_posts:
        url = post["url"]
        topics = extract_topics(post.get("summary", post["title"]))
        new_topics[url] = topics

        print(f"    [topic]: {post['title'][:30]}...")

        G.add_node(url, type="Article", title=post["title"], author=post["author"],
                   summary=post.get("summary", ""), collected_at=post.get("collected_at", ""),
                   points=post.get("points", 0), topics=topics)

        author = post["author"]
        if not G.has_node(author):
            G.add_node(author, type="Author", name=author)
        G.add_edge(url, author, relation="WRITTEN_BY")

        for topic in topics:
            if not G.has_node(topic):
                G.add_node(topic, type="Topic", name=topic)
            G.add_edge(url, topic, relation="TAGGED")

    print("(2/4) [Embedding Generating]")
    for post in new_posts:
        url = post["url"]
        print(f"    [embedding]: {post['title'][:30]}...")
        emb = get_embedding(post.get("summary", post["title"]))
        new_embs[url] = emb
        G.nodes[url]["embedding"] = emb

    print("(3/4) [SIMILAR_TO edge adding]")
    all_embs = {**existing_embs, **new_embs}
    for url_a in new_embs:
        for url_b, emb_b in all_embs.items():
            if url_a == url_b:
                continue
            sim = cosine_similarity(new_embs[url_a], emb_b)
            if sim >= SIMILAR_THRESHOLD:
                G.add_edge(url_a, url_b, relation="SIMILAR_TO", weight=round(sim, 4))
                G.add_edge(url_b, url_a, relation="SIMILAR_TO", weight=round(sim, 4))
                print(f"    {G.nodes[url_a]['title'][:20]} <-> {G.nodes[url_b]['title'][:20]} ({sim:.2f})")

    print("(4/4) [REFERENCES edge adding]")
    all_topics = {**existing_topics, **new_topics}
    for url_a in new_topics:
        for url_b, topics_b in all_topics.items():
            if url_a == url_b:
                continue
            common = set(new_topics[url_a]) & set(topics_b)
            if len(common) >= REFERENCE_MIN_OVERLAP:
                G.add_edge(url_a, url_b, relation="REFERENCES", keywords=list(common))
                print(f"    {G.nodes[url_a]['title'][:20]} -> {G.nodes[url_b]['title'][:20]} {common}")

    return G

def save_graph(G: nx.DiGraph) -> None:
    data = nx.node_link_data(G)
    os.makedirs(os.path.dirname(GRAPH_PATH), exist_ok=True)

    with open(GRAPH_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[saved]: {GRAPH_PATH}")
    print(f"    nodes: {G.number_of_nodes()} | edges: {G.number_of_edges()}")

def load_graph() -> nx.DiGraph:
    with open(GRAPH_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return nx.node_link_graph(data, edges="links")