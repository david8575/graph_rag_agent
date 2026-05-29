import json
import os
from pyvis.network import Network
import networkx as nx

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRAPH_PATH = os.path.join(BASE_DIR, "data", "graph.json")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "graph.html")

NODE_COLORS = {
    "Article": "#4A90D9",
    "Author":  "#27AE60",
    "Topic":   "#E67E22",
}

EDGE_COLORS = {
    "WRITTEN_BY":  "#95A5A6",
    "TAGGED":      "#BDC3C7",
    "SIMILAR_TO":  "#E74C3C",
    "REFERENCES":  "#9B59B6",
}


def load_graph() -> nx.DiGraph:
    with open(GRAPH_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return nx.node_link_graph(data, edges="links")


def build_vis(G: nx.DiGraph) -> Network:
    net = Network(height="800px", width="100%", directed=True, bgcolor="#1a1a2e", font_color="white")
    net.barnes_hut(gravity=-8000, central_gravity=0.3, spring_length=150)

    for node_id, data in G.nodes(data=True):
        ntype = data.get("type", "")
        color = NODE_COLORS.get(ntype, "#888888")

        if ntype == "Article":
            label = data.get("title", node_id)[:30]
            title = f"<b>{data.get('title', '')}</b><br>{node_id}<br>points: {data.get('points', 0)}"
            size = 20
        elif ntype == "Author":
            label = data.get("name", node_id)
            title = f"Author: {data.get('name', node_id)}"
            size = 15
        else:
            label = data.get("name", node_id)
            title = f"Topic: {data.get('name', node_id)}"
            size = 10

        net.add_node(node_id, label=label, color=color, title=title, size=size)

    for src, dst, data in G.edges(data=True):
        relation = data.get("relation", "")
        color = EDGE_COLORS.get(relation, "#555555")
        width = 3 if relation in ("SIMILAR_TO", "REFERENCES") else 1
        net.add_edge(src, dst, color=color, title=relation, width=width)

    return net


def add_legend(html: str) -> str:
    legend = """
<div style="position:fixed;top:20px;right:20px;background:#16213e;padding:16px;border-radius:8px;color:white;font-family:sans-serif;font-size:13px;z-index:999;border:1px solid #444;">
  <b>노드</b><br>
  <span style="color:#4A90D9">●</span> Article &nbsp;
  <span style="color:#27AE60">●</span> Author &nbsp;
  <span style="color:#E67E22">●</span> Topic<br><br>
  <b>엣지</b><br>
  <span style="color:#95A5A6">—</span> WRITTEN_BY<br>
  <span style="color:#BDC3C7">—</span> TAGGED<br>
  <span style="color:#E74C3C">—</span> SIMILAR_TO<br>
  <span style="color:#9B59B6">—</span> REFERENCES
</div>
"""
    return html.replace("</body>", legend + "</body>")


def visualize():
    G = load_graph()
    net = build_vis(G)
    net.save_graph(OUTPUT_PATH)

    with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
        html = f.read()
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(add_legend(html))

    print(f"[visualize] nodes: {G.number_of_nodes()} | edges: {G.number_of_edges()}")
    print(f"[visualize] saved → {OUTPUT_PATH}")
