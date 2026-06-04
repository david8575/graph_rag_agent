import json
import os
import networkx as nx
import numpy as np
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_ollama import OllamaLLM, OllamaEmbeddings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRAPH_PATH = os.path.join(BASE_DIR, "data", "graph.json")
OLLAMA_URL = "http://localhost:11434"
CHAT_MODEL = "gemma4:e4b"

llm = OllamaLLM(model=CHAT_MODEL, base_url=OLLAMA_URL)
embed_model = OllamaEmbeddings(model="mxbai-embed-large", base_url=OLLAMA_URL)

class QAState(TypedDict):
    question: str
    question_type: str # search, recommand, explain
    keywords: List[str]
    retrieved_articles: List[str]
    answer: str

def load_graph() -> nx.DiGraph:
    with open(GRAPH_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    return nx.node_link_graph(data, edges="links")

def ollama(prompt: str) -> str:
    return llm.invoke(prompt).strip()

# node1: 질문 유형 분류
def classify_question(state: QAState) -> QAState:
    prompt = f"""
        반드시 아래 셋 중 하나만 출력하세요.
        - search: 특정 기술/주제 글을 찾을 때
        - recommand: 관련 글 추천을 요청할 때
        - explain: 개념/기술 설명을 요청할 때

        질문: {state["question"]}

        유형:
    """

    raw = ollama(prompt=prompt).lower()

    if "recommmand" in raw:
        q_type = "recommand"
    elif "explain" in raw:
        q_type = "explain"
    else:
        q_type = "search"

    print(f"[classify] {q_type}")

    return {**state, "question_type": q_type}

# node2: 키워드 추출
def extract_keywords(state: QAState) -> QAState:
    prompt = f"""
        다음 질문에서 검색 키워드를 1~3개 추출하세요.
        쉼표로 구분하여 키워드만 나열하세요.
        
        질문: {state["question"]}

        키워드: 
    """

    raw = ollama(prompt=prompt)
    keywords = [k.strip() for k in raw.split(",") if k.strip()][:3]

    print(f"[keywords] {keywords}")

    return {**state, "keywords": keywords}

# node 3: 그래프 RAG 검색
def search_graph(state: QAState) -> QAState:
    G = load_graph()
    keywords = state["keywords"]
    question_type = state["question_type"]
    question = state["question"]

    matched_urls = set()

    # 방법 1: Topic 이름 키워드 매칭
    for node_id, data in G.nodes(data=True):
        if data.get("type") == "Topic":
            topic_name = data.get("name", "")
            for kw in keywords:
                if kw in topic_name or topic_name in kw:
                    for pred in G.predecessors(node_id):
                        if G.nodes[pred].get("type") == "Article":
                            matched_urls.add(pred)

    # 방법 2: 질문 임베딩 기반 유사도 검색
    q_embedding = embed_model.embed_query(question)
    q_vec = np.array(q_embedding)

    scored = []
    for node_id, data in G.nodes(data=True):
        if data.get("type") == "Article" and "embedding" in data:
            a_vec = np.array(data["embedding"])
            sim = float(np.dot(q_vec, a_vec) / (np.linalg.norm(q_vec) * np.linalg.norm(a_vec)))
            scored.append((node_id, sim))

    scored.sort(key=lambda x: x[1], reverse=True)
    for url, sim in scored[:3]:
        print(f"    [vector] {G.nodes[url]['title'][:30]} ({sim:.2f})")
        matched_urls.add(url)

    # recommend: SIMILAR_TO 이웃 확장
    if question_type == "recommand":
        expanded = set(matched_urls)
        for url in matched_urls:
            for neighbor in G.successors(url):
                if G.nodes[neighbor].get("type") == "Article":
                    edge = G.get_edge_data(url, neighbor, {})
                    if edge.get("relation") == "SIMILAR_TO":
                        expanded.add(neighbor)
        matched_urls = expanded

    articles = []
    for url in matched_urls:
        d = G.nodes[url]
        articles.append({
            "title": d.get("title", ""),
            "url": url,
            "summary": d.get("summary", ""),
            "author": d.get("author", ""),
            "points": d.get("points", 0)
        })

    articles.sort(key=lambda x: x["points"], reverse=True)

    print(f"[search] {len(articles)} found")
    for a in articles[:5]:
        print(f"    - {a['title'][:40]}")

    return {**state, "retrieved_articles": articles[:5]}


# 노드 4: 답변 생성
def generate_answer(state: QAState) -> QAState:
    articles = state["retrieved_articles"]

    if not articles:
        return {**state, "answer": "관련된 글을 찾을 수 없습니다."}
    
    context = "\n\n".join([
        f"제목: {a['title']}\n요약: {a['summary']}\nURL: {a['url']}"
        for a in articles
    ])

    prompt = f"""
        아래 기술 뉴스들을 참고해서 질문에 3~5 문장으로 답변하세요.

        참고글: {context}

        답변: 
    """

    answer = ollama(prompt)

    return {**state, "answer": answer}

# LangGraph 구성

def build_workflow():
    wf = StateGraph(QAState)

    wf.add_node("classify", classify_question)
    wf.add_node("keywords", extract_keywords)
    wf.add_node("search", search_graph)
    wf.add_node("answer", generate_answer)

    wf.set_entry_point("classify")
    wf.add_edge("classify", "keywords")
    wf.add_edge("keywords", "search")
    wf.add_edge("search", "answer")
    wf.add_edge("answer", END)

    return wf.compile()

def ask(question: str) -> str:
    app = build_workflow()

    result = app.invoke({
        "question": question,
        "question_type": "",
        "keywords": [],
        "retrieved_articles": [],
        "answer": ""
    })

    return {
        "answer": result["answer"],
        "retrieved_articles": [a["url"] for a in result["retrieved_articles"]]
    }
