# app.py (Flask + offline HuggingFace chatbot using Flan-T5 + focused on Kho-Kho only)
import os
import json
import re
import pytz
from datetime import datetime
from flask import Flask, render_template, request, jsonify, session, url_for
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
from pytz import timezone as pytz_timezone, UnknownTimeZoneError
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline

# --- Configuration ---
SECRET_KEY = os.getenv("KHO_KHO_BOT_SECRET", "replace-with-secure-random-key")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "google/flan-t5-base"
JSON_DATA_FILE = os.getenv("KHO_KHO_DATA_FILE", "kho-kho.json")
MAX_TOKENS = int(os.getenv("KHO_KHO_MAX_TOKENS", "128"))

app = Flask(__name__)
app.secret_key = SECRET_KEY

# --- Utility Functions ---
def get_greeting():
    user_tz_name = session.get("timezone", "UTC")
    try:
        user_tz = pytz.timezone(user_tz_name)
    except pytz.UnknownTimeZoneError:
        user_tz = pytz.utc
    user_time = datetime.now(user_tz)
    hour = user_time.hour
    if hour < 12:
        return "Good morning"
    elif hour < 17:
        return "Good afternoon"
    else:
        return "Good evening"

# --- Data Loading ---
def load_documents(path: str):
    with open(path, encoding="utf-8") as f:
        records = json.load(f).get("conversations", [])
    return [Document(page_content=f"Q: {q}\nA: {a}") for q, a in records]

documents = load_documents(JSON_DATA_FILE)

# --- Embeddings & LLM Setup ---
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
_model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL)
_pipe = pipeline(
    "text2text-generation",
    model=_model,
    tokenizer=_tokenizer,
    max_new_tokens=MAX_TOKENS,
)
llm = HuggingFacePipeline(pipeline=_pipe)

# --- Vector Store & Memory ---
vectorstore = FAISS.from_documents(documents, embeddings)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
retriever = vectorstore.as_retriever()

# --- Flask Routes ---
@app.route("/")
def index():
    return render_template("index.html", initial_message="", video=None)

@app.route("/timezone", methods=["POST"])
def set_timezone():
    tz = request.json.get("timezone", "")
    try:
        session["timezone"] = tz if tz else "UTC"
    except UnknownTimeZoneError:
        session["timezone"] = "UTC"
    return jsonify({"status": "ok"})

@app.route("/greeting", methods=["GET"])
def greeting():
    return jsonify({"greeting": get_greeting()})

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "").strip()
    if not user_input:
        return jsonify({"message": "Please say something about Kho-Kho!"})
    print(session)
    if "username" not in session:
        m = re.match(r"my name is\s+([A-Za-z]{2,20})", user_input, re.IGNORECASE)
        if m:
            session["username"] = m.group(1).capitalize()
            return jsonify({"message": f"Nice to meet you, {session['username']}! I can answer your Kho-Kho related questions."})
        else:
            return jsonify({"message": "Please enter your name in the format: My name is <YourName>."})
    name = session.get("username", "friend")
    if re.search(r"\b(bye|exit|quit|goodbye)\b", user_input, re.IGNORECASE):
        session.clear()
        return jsonify({"message": "Goodbye! Your session has ended."})

    docs = retriever.get_relevant_documents(user_input)
    if not docs:
        fallback = (
            f"Sorry {name}, I only answer Kho-Kho related queries. "
            "Try: 'How do you play kho-kho?', 'What are the rules for kho-kho?'"
        )
        return jsonify({"message": fallback})

    context = "\n\n".join(doc.page_content for doc in docs)
    enforced = "Only discuss Kho-Kho. Redirect back to Kho-Kho if off-topic.\n"
    need_video = bool(re.search(r"\b(rules|how to play|video)\b", user_input, re.I))
    show_video = need_video and not session.get("video_sent", False)
    if show_video:
        session["video_sent"] = True
    elif re.search(r"\b(show|play|video)\b", user_input, re.I):
        show_video = True

    implicit = "Context: rules and gameplay details.\n" if need_video else ""
    prompt = (
        f"You are Khokho-Bot answering {name} about Kho-Kho.\n"
        f"{enforced}\n"
        f"{implicit}\n"
        f"Context:\n{context}\n"
        f"Question: {user_input}\nAnswer:"
    )
    raw = llm.invoke(prompt).strip()
    answer = (
        f"I didn't catch that, {name}. Can you ask about Kho-Kho again?"
        if len(raw) < 5 or raw.lower() in ["a).", "b)."]
        else raw
    )

    video_url = url_for('static', filename='kho-kho.mp4') if show_video else None
    return jsonify({"message": answer, "video": video_url})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)), debug=False)
