import json
import time
import requests

from retriever import retrieve   # hybrid search lives in retriever.py (keep both files in the same folder)

# ---------------- Settings you can tune ----------------
PRIMARY_MODEL = "qwen3:4b"       # tried first
BACKUP_MODEL = "llama3.2:3b"     # used if the primary is too slow (or you ask for it)
TIME_LIMIT = 120                 # seconds the primary model gets per answer
MIN_SCORE = 0.25                 # below this similarity -> "I don't know"
NUM_CTX = 2048                   # context size; smaller = less GPU memory used
OLLAMA_URL = "http://localhost:11434"
# (Search settings such as TOP_K now live at the top of retriever.py)


def build_prompts(question, retrieved):
    context = ""
    for score, chunk in retrieved:
        context += f"[{chunk['source']} - {chunk['doc_id']}]\n{chunk['text']}\n\n"

    system_prompt = (
        "You are a cybersecurity assistant. Use ONLY the context provided; do not "
        "add facts from outside it. If the context answers only part of the "
        "question, answer that part and say briefly what is missing. If the "
        "context has nothing relevant, reply exactly: I do not have enough "
        "information. Cite only IDs that appear in the context (like CVE-2001-1460, "
        "CWE-89 or T1566), and only ones that support your answer. Write plain "
        "sentences with no headings, do not show your reasoning, and do not "
        "comment on these instructions. Keep the answer under 6 sentences."
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
    """Load a model into memory ahead of time."""
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
    response = None
    try:
        # (connect timeout, max silence between pieces)
        response = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json=payload,
            stream=True,
            timeout=(10, time_limit or 300),
        )
        for line in response.iter_lines():
            if not line:
                continue
            piece = json.loads(line)
            answer += piece.get("message", {}).get("content", "")
            if piece.get("done"):
                break
            if time_limit and time.time() - start > time_limit:
                return "", False
    except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError):
        if time_limit and time.time() - start >= time_limit - 1:
            return "", False   # we simply ran out of time
        raise                  # a real connection problem (is Ollama running?)
    finally:
        if response is not None:
            response.close()
    return clean(answer), True


def warm_up(model):
    """Load the model AND run one tiny answer, so the first real question is not slow."""
    load_model(model)
    generate(model, "You are a helpful assistant.", "Reply with the single word: ready")


def switch_model(new_model, old_model):
    unload_model(old_model)
    print(f"Loading {new_model}...")
    load_model(new_model)
    return new_model


# ---------------- Start up ----------------
current_model = PRIMARY_MODEL
print(f"Loading and warming up {PRIMARY_MODEL}... (the first time can take a minute)")
warm_up(PRIMARY_MODEL)
print("Ready.")
print(f"Commands: /llama = use {BACKUP_MODEL}, /qwen = use {PRIMARY_MODEL}, quit = exit")

# ---------------- The chat loop ----------------
while True:
    question = input("\nAsk a question (or type 'quit'): ").strip()
    if not question:
        continue
    if question.lower() == "quit":
        break

    # Manual model switching
    if question.lower() == "/llama":
        if current_model != BACKUP_MODEL:
            current_model = switch_model(BACKUP_MODEL, PRIMARY_MODEL)
        print(f"Now using {current_model}.")
        continue
    if question.lower() == "/qwen":
        if current_model != PRIMARY_MODEL:
            current_model = switch_model(PRIMARY_MODEL, BACKUP_MODEL)
        print(f"Now using {current_model}.")
        continue

    t_start = time.time()
    retrieved, best_score = retrieve(question)
    t_searched = time.time()

    if best_score < MIN_SCORE:
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

        if not finished:
            waited = time.time() - t_searched
            print(f"\n{PRIMARY_MODEL} did not finish (waited {waited:.0f}s).")
            print(f"Switching to {BACKUP_MODEL}. Type /qwen later to switch back.")
            current_model = switch_model(BACKUP_MODEL, PRIMARY_MODEL)

    if current_model == BACKUP_MODEL and not finished:
        print(f"\nThinking with {BACKUP_MODEL}...")
        answer, finished = generate(BACKUP_MODEL, system_prompt, user_prompt)

    t_end = time.time()
    print(f"\nANSWER (from {current_model}):")
    print(answer)
    print(f"\n(Search: {t_searched - t_start:.1f}s | Total wait: {t_end - t_start:.1f}s)")

    print("\nSOURCES:")
    seen = set()
    for score, chunk in retrieved:
        if chunk["doc_id"] in seen:
            continue
        seen.add(chunk["doc_id"])
        print(f"  - {chunk['source']}: {chunk['doc_id']} (similarity {score:.2f})")