import os
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUMMARIZED_PATH = os.path.join(BASE_DIR, "data", "summarized.json")

GMAIL_USER = os.getenv("GMAIL_USER")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD", "").replace(" ", "")
GMAIL_TO = os.getenv("GMAIL_TO")


def load_recent_posts(n: int = 10) -> list:
    with open(SUMMARIZED_PATH, "r", encoding="utf-8") as f:
        posts = json.load(f)
    posts.sort(key=lambda x: x.get("points", 0), reverse=True)
    return posts[:n]


def build_email_body(posts: list) -> str:
    today = datetime.now().strftime("%Y년 %m월 %d일")
    lines = [f"<h2>📰 GeekNews 브리핑 - {today}</h2>"]

    for i, post in enumerate(posts, 1):
        lines.append(f"""
        <div style="margin-bottom:24px;">
            <h3>{i}. <a href="{post['url']}">{post['title']}</a></h3>
            <p>{post.get('summary', '')}</p>
            <small>👤 {post.get('author', '')} | ⭐ {post.get('points', 0)}pts</small>
        </div>
        """)

    return "\n".join(lines)


def send_briefing():
    posts = load_recent_posts()

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"[GeekNews] 오늘의 기술 브리핑 {datetime.now().strftime('%m/%d')}"
    msg["From"] = GMAIL_USER
    msg["To"] = GMAIL_TO

    body = build_email_body(posts)
    msg.attach(MIMEText(body, "html"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(GMAIL_USER, GMAIL_APP_PASSWORD)
        server.sendmail(GMAIL_USER, GMAIL_TO, msg.as_string())

    print(f"[email] sent to {GMAIL_TO} ({len(posts)} posts)")


if __name__ == "__main__":
    send_briefing()
