"""
Scores how well each search strategy finds the right records.
No AI model is needed for this one, so it runs in seconds.
Reads eval_questions.json and writes retrieval_eval_results.csv.
"""
import csv
import json

import numpy as np

from retriever import TOP_K, bm25, chunks, embedder, hybrid_candidates, index, retrieve, tokenize

K = TOP_K   # we judge the same number of chunks the chatbot gives the model

with open("eval_questions.json", "r", encoding="utf-8") as f:
    questions = json.load(f)

known_ids = {c["doc_id"] for c in chunks}


# ---------------- The four strategies (each returns a list of record IDs) ----------------
def meaning_only(question):
    vector = embedder.encode([question], normalize_embeddings=True)
    vector = np.array(vector, dtype="float32")
    _, positions = index.search(vector, K)
    return [chunks[int(p)]["doc_id"] for p in positions[0]]


def keyword_only(question):
    scores = bm25.get_scores(tokenize(question))
    order = np.argsort(scores)[::-1][:K]
    return [chunks[int(p)]["doc_id"] for p in order]


def hybrid(question):
    return [chunks[pos]["doc_id"] for _, _, pos in hybrid_candidates(question)[:K]]


def final_system(question):
    results, _ = retrieve(question)
    return [chunk["doc_id"] for _, chunk in results]


SYSTEMS = {
    "1. Meaning search only": meaning_only,
    "2. Keyword search only": keyword_only,
    "3. Hybrid (3 searches)": hybrid,
    "4. Hybrid + balanced (final)": final_system,
}


def distinct(ids):
    seen, out = set(), []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def score(retrieved, relevant):
    """Returns (hit, precision, reciprocal_rank) for one question."""
    ids = distinct(retrieved)
    found = [i for i in ids if i in relevant]
    hit = 1 if found else 0
    precision = len(found) / len(ids) if ids else 0.0
    rr = 0.0
    for rank, i in enumerate(ids, start=1):
        if i in relevant:
            rr = 1.0 / rank
            break
    return hit, precision, rr


# ---------------- Check the test set against your knowledge base ----------------
usable = []
for q in questions:
    if not q["relevant"]:
        continue   # out-of-scope questions are not part of retrieval scoring
    valid = [i for i in q["relevant"] if i in known_ids]
    missing = [i for i in q["relevant"] if i not in known_ids]
    if missing:
        print(f"Warning: {q['id']} lists IDs that are not in your knowledge base: {missing}")
    if valid:
        usable.append({**q, "relevant": valid})
    else:
        print(f"Skipping {q['id']}: none of its relevant records exist in your knowledge base.")

print(f"\nScoring {len(usable)} questions with K = {K} chunks each.\n")

# ---------------- Run every strategy on every question ----------------
rows = []
totals = {name: {"hit": 0, "precision": 0.0, "rr": 0.0} for name in SYSTEMS}
by_category = {}   # category -> hits for the final system

for q in usable:
    for name, strategy in SYSTEMS.items():
        retrieved = distinct(strategy(q["question"]))
        hit, precision, rr = score(retrieved, set(q["relevant"]))
        totals[name]["hit"] += hit
        totals[name]["precision"] += precision
        totals[name]["rr"] += rr
        rows.append([q["id"], q["category"], name, " ".join(retrieved), hit, f"{precision:.2f}", f"{rr:.2f}"])
        if name.startswith("4."):
            stats = by_category.setdefault(q["category"], [0, 0])
            stats[0] += hit
            stats[1] += 1
            if not hit:
                print(f"MISS (final system) {q['id']}: {q['question']}")
                print(f"     wanted one of: {q['relevant'][:6]}   got: {retrieved}")

# ---------------- Results ----------------
n = len(usable)
print("\n" + "=" * 72)
print(f"{'Strategy':<32}{'Hit@K':>10}{'Precision@K':>14}{'MRR':>10}")
print("=" * 72)
for name, t in totals.items():
    print(f"{name:<32}{t['hit'] / n:>10.2f}{t['precision'] / n:>14.2f}{t['rr'] / n:>10.2f}")
print("=" * 72)

print("\nFinal system, hit rate by kind of question:")
for category, (hits, count) in by_category.items():
    print(f"  {category:<12} {hits}/{count}")

with open("retrieval_eval_results.csv", "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow(["question_id", "category", "strategy", "retrieved_records", "hit", "precision", "reciprocal_rank"])
    writer.writerows(rows)
print("\nSaved retrieval_eval_results.csv")
print("Settings used: TOP_K, CAP_RECORDS_PER_TYPE, MARGIN and RRF_K from retriever.py (write them in your report).")