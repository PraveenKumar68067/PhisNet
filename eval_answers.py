"""
Runs every question in eval_questions.json through the chatbot and saves the answers,
with automatic citation checks and empty columns for you to score by hand.

Modes:
  rag   - the real chatbot (search + model)
  norag - the same model with NO search, as your baseline
Nothing here changes your chatbot. Ollama must be running.
"""
import csv
import json
import re
import time

import requests

from retriever import chunks, retrieve

# ---------------- Settings ----------------
MODELS = {"": "llama3.2:3b", "llama": "llama3.2:3b", "qwen": "qwen3:4b"}
NUM_CTX = 2048
MIN_SCORE = 0.25            # same "I don't know" threshold as the chatbot
OLLAMA_URL = "http://localhost:11434"

ID_PATTERN = re.compile(r"CVE-\d{4}-\d{4,7}|CWE-\d+|T\d{4}(?:\.\d{3})?")
KNOWN_IDS = {c["doc_id"] for c in chunks}

# Keep this prompt in sync with rag_chat_v5.py
RAG_SYSTEM = (
    "You are a cybersecurity assistant. Use ONLY the context provided; do not "
    "add facts from outside it. If the context answers only part of the "
    "question, answer that part and say briefly what is missing. If the "
    "context has nothing relevant, reply exactly: I do not have enough "
    "information. Cite only IDs that appear in the context (like CVE-2001-1460, "
    "CWE-89 or T1566), and only ones that support your answer. Write plain "
    "sentences with no headings, do not show your reasoning, and do not "
    "comment on these instructions. Keep the answer under 6 sentences."
)
NORAG_SYSTEM = (
    "You are a cybersecurity assistant. Answer in under 6 sentences and cite the "
    "relevant CVE, CWE or MITRE ATT&CK IDs. Write plain sentences with no headings."
)


def ask(model, system_prompt, user_prompt):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "keep_alive": "10m",
        "options": {"num_ctx": NUM_CTX, "temperature": 0, "num_predict": 400},
    }
    if model.startswith("qwen3"):
        payload["think"] = False
    response = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=900)
    data = response.json()
    if "message" not in data:
        raise RuntimeError(data.get("error", "Unexpected reply from Ollama"))
    text = data["message"]["content"]
    if "</think>" in text:
        text = text.split("</think>")[-1]
    return text.strip()


def is_refusal(text):
    lowered = text.lower()
    return "do not have enough information" in lowered or "could not find relevant" in lowered


# ---------------- Setup ----------------
with open("eval_questions.json", "r", encoding="utf-8") as f:
    questions = json.load(f)

choice = input("Model: press Enter for llama3.2:3b, or type 'qwen' for qwen3:4b (slow!): ")
model = MODELS.get(choice.strip().lower(), MODELS[""])
mode = input("Mode: press Enter for 'rag', or type 'norag' for the no-search baseline: ").strip().lower()
mode = "norag" if mode == "norag" else "rag"
print(f"\nRunning {len(questions)} questions with {model} in {mode} mode.\n")

rows = []
stats = {"in_scope": 0, "citation_hit": 0, "fabricated": 0, "not_in_context": 0,
         "false_refusals": 0, "out_scope": 0, "correct_refusals": 0, "seconds": 0.0}

for q in questions:
    start = time.time()
    retrieved_ids = []

    if mode == "rag":
        retrieved, best_score = retrieve(q["question"])
        retrieved_ids = sorted({chunk["doc_id"] for _, chunk in retrieved})
        if best_score < MIN_SCORE:
            answer = "I could not find relevant information in my knowledge base."
        else:
            context = "".join(
                f"[{chunk['source']} - {chunk['doc_id']}]\n{chunk['text']}\n\n" for _, chunk in retrieved
            )
            answer = ask(model, RAG_SYSTEM, f"Context:\n{context}\nQuestion: {q['question']}")
    else:
        answer = ask(model, NORAG_SYSTEM, q["question"])

    seconds = time.time() - start
    cited = sorted(set(ID_PATTERN.findall(answer)))
    fabricated = [i for i in cited if i not in KNOWN_IDS]            # IDs that are not in your knowledge base
    not_in_context = [i for i in cited if mode == "rag" and i not in retrieved_ids]
    relevant = set(q["relevant"])
    citation_hit = bool(relevant & set(cited)) if relevant else None
    refused = is_refusal(answer)

    stats["seconds"] += seconds
    if relevant:
        stats["in_scope"] += 1
        stats["citation_hit"] += 1 if citation_hit else 0
        stats["false_refusals"] += 1 if refused else 0
    else:
        stats["out_scope"] += 1
        stats["correct_refusals"] += 1 if refused else 0
    stats["fabricated"] += 1 if fabricated else 0
    stats["not_in_context"] += 1 if not_in_context else 0

    print(f"{q['id']}  {seconds:5.1f}s  cited={cited}  refused={refused}")
    rows.append([
        q["id"], q["category"], q["question"], answer, " ".join(retrieved_ids), " ".join(cited),
        "" if citation_hit is None else int(citation_hit), " ".join(fabricated), " ".join(not_in_context),
        int(refused), f"{seconds:.1f}", "", "",
    ])

# ---------------- Summary and files ----------------
name = f"answers_{model.split(':')[0]}_{mode}"
with open(f"{name}.csv", "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow([
        "question_id", "category", "question", "answer", "retrieved_records", "cited_ids",
        "cited_a_relevant_id", "cited_ids_not_in_knowledge_base", "cited_ids_not_in_retrieved_context",
        "refused", "seconds", "faithfulness_1to5 (fill in)", "relevance_1to5 (fill in)",
    ])
    writer.writerows(rows)

total = len(questions)
print("\n" + "=" * 64)
print(f"Model {model}, mode {mode}")
print("=" * 64)
print(f"Average time per answer:               {stats['seconds'] / total:.1f} s")
print(f"Answers that cited a relevant ID:      {stats['citation_hit']}/{stats['in_scope']}")
print(f"Answers citing IDs NOT in your data:   {stats['fabricated']}/{total}   (possible made-up IDs)")
if mode == "rag":
    print(f"Answers citing IDs not in context:     {stats['not_in_context']}/{total}")
print(f"Wrongly refused (in-scope questions):  {stats['false_refusals']}/{stats['in_scope']}")
print(f"Correctly refused (out-of-scope):      {stats['correct_refusals']}/{stats['out_scope']}")
print("=" * 64)
print(f"Saved {name}.csv. Open it in Excel and fill in the two score columns by hand.")