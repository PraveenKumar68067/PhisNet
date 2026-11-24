import json
import re
import time
import numpy as np
import faiss
import requests
from sentence_transformers import SentenceTransformer

# ---------------- Settings you can tune ----------------
PRIMARY_MODEL = "qwen3:4b"       # tried first
BACKUP_MODEL = "llama3.2:3b"     # used if the primary is too slow (or you skip it)
TIME_LIMIT = 60                  # seconds the primary model gets per answer
TOP_K = 4                        # chunks retrieved per question
MIN_SCORE = 0.25                 # below this similarity -> "I don't know"
NUM_CTX = 2048                   # context size; smaller = less GPU memory used
OLLAMA_URL = "http://localhost:11434"

# ---------------- Load the search engine ----------------
index = faiss.read_index("index.faiss")
with open("chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)
embedder = SentenceTransformer("all-MiniLM-L6-v2")


def retrieve(question):
    """Find the most relevant chunks for a question."""
    q_vector = embedder.encode([question], normalize_embeddings=True)
    q_vector = np.array(q_vector, dtype="float32")
    scores, positions = index.search(q_vector, TOP_K)
    return [(float(s), chunks[p]) for s, p in zip(scores[0], positions[0])]


def build_prompts(question, retrieved):
    context = ""
    for score, chunk in retrieved:
        context += f"[{chunk['source']} - {chunk['doc_id']}]\n{chunk['text']}\n\n"

    system_prompt = (
        "You are a cybersecurity assistant. Use ONLY the context provided; do not "
        "add facts from outside it. If the context answers only part of the "
        "question, answer that part and say clearly what is missing. If the "
        "context has nothing relevant, reply exactly: I do not have enough "
        "information. Mention the CVE or Technique IDs you used. Do not show your "
        "reasoning. Keep the answer under 6 sentences."
    )
    user_prompt = f"Context:\n{context}\nQuestion: {question}"
    return system_prompt, user_prompt


def clean(text):
    """
    Remove reasoning text. Some models print their reasoning and then a closing
    </think> tag with no opening tag, so keep only what comes after the last one.
    """
    if "</think>" in text:
        text = text.split("</think>")[-1]
    return text.strip()


def load_model(model):
    """Load a model into memory ahead of time, so loading is not counted as 'slow'."""
    requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": model, "keep_alive": "30m", "options": {"num_ctx": NUM_CTX}},
        timeout=600,
    )


def unload_model(model):
    """Free the model's memory so two models never compete for your 4GB VRAM."""
    try:
        requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": model, "keep_alive": 0},
            timeout=30,
        )
    except requests.RequestException:
        pass


def generate(model, system_prompt, user_prompt, time_limit=None):
    """
    Ask a model for an answer, receiving it piece by piece.
    Returns (answer, finished). finished is False if we ran out of time.
    """
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": True,
        "keep_alive": "30m",
        "options": {"num_ctx": NUM_CTX},
    }
    if model.startswith("qwen3"):
        payload["think"] = False   # skip slow "thinking" mode

    start = time.time()
    answer = ""
    # (connect timeout, max silence between pieces)
    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json=payload,
        stream=True,
        timeout=(10, time_limit or 300),
    )
    try:
        for line in response.iter_lines():
            if not line:
                continue
            piece = json.loads(line)
            answer += piece.get("message", {}).get("content", "")
            if piece.get("done"):
                break
            if time_limit and time.time() - start > time_limit:
                return "", False
    finally:
        response.close()
    return clean(answer), True


# ---------------- Start up ----------------
current_model = PRIMARY_MODEL
print(f"Loading {PRIMARY_MODEL}... (the first time can take a minute)")
load_model(PRIMARY_MODEL)
print("Ready.")

# ---------------- The chat loop ----------------
while True:
    question = input("\nAsk a question (or type 'quit'): ")
    if question.lower() == "quit":
        break

    retrieved = retrieve(question)

    if retrieved[0][0] < MIN_SCORE:
        print("\nI could not find relevant information in my knowledge base.")
        continue

    system_prompt, user_prompt = build_prompts(question, retrieved)
    answer, finished = "", False

    if current_model == PRIMARY_MODEL:
        print(f"\nThinking with {PRIMARY_MODEL}... (press Ctrl+C to switch to {BACKUP_MODEL})")
        try:
            answer, finished = generate(PRIMARY_MODEL, system_prompt, user_prompt, TIME_LIMIT)
        except KeyboardInterrupt:
            print("\nSkipped by you.")
        except requests.exceptions.ReadTimeout:
            print("\nToo slow.")

        if not finished:
            print(f"Switching to {BACKUP_MODEL} for the rest of this session...")
            unload_model(PRIMARY_MODEL)
            current_model = BACKUP_MODEL

    if current_model == BACKUP_MODEL and not finished:
        print(f"\nThinking with {BACKUP_MODEL}...")
        answer, finished = generate(BACKUP_MODEL, system_prompt, user_prompt)

    print(f"\nANSWER (from {current_model}):")
    print(answer)
    print("\nSOURCES:")
    for score, chunk in retrieved:
        print(f"  - {chunk['source']}: {chunk['doc_id']} (similarity {score:.2f})")
