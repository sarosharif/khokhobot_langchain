# app.py (Flask + offline HuggingFace chatbot using Flan-T5 + focused on Kho-Kho only)
import os
import json
import re
from datetime import datetime
from flask import Flask, render_template, request, jsonify, session, url_for
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline

# --- Configuration ---
SECRET_KEY = os.getenv("KHO_KHO_BOT_SECRET", "replace-with-secure-random-key")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "google/flan-t5-small"
JSON_DATA_FILE = os.getenv("KHO_KHO_DATA_FILE", "kho-kho.json")
MAX_TOKENS = int(os.getenv("KHO_KHO_MAX_TOKENS", "128"))

app = Flask(__name__)
app.secret_key = SECRET_KEY

# --- Utility Functions ---
def get_greeting():
    now_hour = datetime.now().hour
    if now_hour < 12:
        return "Good morning"
    if now_hour < 17:
        return "Good afternoon"
    return "Good evening"

# --- Data Loading ---
def load_documents(path: str):
    """Load Q&A pairs from JSON into Document list."""
    with open(path, encoding="utf-8") as f:
        records = json.load(f).get("conversations", [])
    return [Document(page_content=f"Q: {q}\nA: {a}") for q, a in records]

documents = load_documents(JSON_DATA_FILE)

# --- Embeddings & LLM Setup ---
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# Initialize the Flan-T5 pipeline
_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
_model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL)
_pipe = pipeline(
    "text2text-generation",
    model=_model,
    tokenizer=_tokenizer,
    max_new_tokens=MAX_TOKENS,
)  # careful: GPU/CPU memory
llm = HuggingFacePipeline(pipeline=_pipe)

# --- Vector Store & Memory ---
vectorstore = FAISS.from_documents(documents, embeddings)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
retriever = vectorstore.as_retriever()

# --- Flask Routes ---
@app.route("/")
def index():
    greeting = get_greeting()
    return render_template(
        "index.html",
        initial_message=f"{greeting}! I'm Khokho-Bot. What's your name?",
        video=None,
    )

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "").strip()
    if not user_input:
        return jsonify({"message": "Please say something about Kho-Kho!"})
    print(session)
    # Name detection (store once)
    if "username" not in session:
        m = re.match(r"my name is\s+([A-Za-z]{2,20})", user_input, re.IGNORECASE)
        if m:
            session["username"] = m.group(1).capitalize()
            return jsonify({"message": f"Nice to meet you, {session['username']}! I can answer your Kho-Kho related questions."})
        else:
            return jsonify({"message": "Please enter your name in the format: My name is <YourName>."})
    name = session.get("username", "friend")
    if re.search(r"\b(bye|exit|quit|goodbye)\b", user_input, re.IGNORECASE):
        session.clear()  # This clears all session data, including username and flags
        return jsonify({"message": "Goodbye! Your session has ended."})
    # Retrieve relevant context
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

    # Check if user has already received the video
    show_video = need_video and not session.get("video_sent", False)

    if show_video:
        session["video_sent"] = True
    elif re.search(r"\b(show|play|video)\b", user_input, re.I):
        show_video = True  # allow re-show on explicit request

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
