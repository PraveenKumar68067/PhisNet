import json
import re
from functools import lru_cache

import faiss
import numpy as np
from nltk.stem import PorterStemmer
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# ---------------- Settings you can tune ----------------
TOP_K = 4                # chunks given to the model per question
SEARCH_DEPTH = 40        # how deep each search looks before the results are combined
CAP_RECORDS_PER_TYPE = 3 # at most this many different records of one kind (CVE / CWE / technique)
MARGIN = 0.15            # a chunk must score within this of the best meaning-score
RRF_K = 60               # standard constant for combining rankings

# ---------------- Load everything once ----------------
index = faiss.read_index("index.faiss")
with open("chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------- Keyword search (BM25) setup ----------------
_stemmer = PorterStemmer()
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does",
    "for", "from", "how", "i", "in", "into", "is", "it", "me", "my", "of",
    "on", "or", "that", "the", "their", "they", "this", "to", "was", "what",
    "when", "where", "which", "who", "why", "with",
}


@lru_cache(maxsize=None)
def _stem(word):
    return _stemmer.stem(word)


def tokenize(text):
    """Lowercase, split into words, drop filler words, and cut words to their stems."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [_stem(w) for w in words if w not in STOPWORDS]


# Search 2 looks at the whole text of every chunk
bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])


def _title_of(chunk):
    """
    The first two lines of a chunk (the ID and the name). Every chunk of a record
    starts with them. CVEs have no real name, so they get none.
    """
    if chunk["type"] == "cve":
        return ""
    return " ".join(chunk["text"].split("\n")[:2])


# Search 3 looks ONLY at names, so "cross-site scripting" favors entries NAMED that
title_bm25 = BM25Okapi([tokenize(_title_of(c)) for c in chunks])


def _keyword_ranking(scores, kind):
    """Rank chunks by keyword score (only those above zero), optionally one kind only."""
    ranking = {}
    for pos in np.argsort(scores)[::-1]:
        pos = int(pos)
        if scores[pos] <= 0:
            break
        if kind is not None and chunks[pos]["type"] != kind:
            continue
        ranking[pos] = len(ranking) + 1
        if len(ranking) == SEARCH_DEPTH:
            break
    return ranking


def hybrid_candidates(question, kind=None):
    """
    Rank chunks with three searches (meaning, keywords, names) and combine them.
    If kind is given ("cwe", "cve" or "technique"), only that kind is considered.
    Returns a list of (fused_score, meaning_score, position), best first.
    """
    q_vector = embedder.encode([question], normalize_embeddings=True)
    q_vector = np.array(q_vector, dtype="float32")

    # Search 1: by meaning
    _, d_positions = index.search(q_vector, len(chunks))
    meaning_rank = {}
    for pos in d_positions[0]:
        pos = int(pos)
        if kind is not None and chunks[pos]["type"] != kind:
            continue
        meaning_rank[pos] = len(meaning_rank) + 1
        if len(meaning_rank) == SEARCH_DEPTH:
            break

    # Search 2: by keywords in the whole text
    keyword_rank = _keyword_ranking(bm25.get_scores(tokenize(question)), kind)

    # Search 3: by keywords in the name only
    title_rank = _keyword_ranking(title_bm25.get_scores(tokenize(question)), kind)

    # Combine: a chunk earns points from each search, more points for a higher rank
    fused = {}
    for ranks in (meaning_rank, keyword_rank, title_rank):
        for pos, rank in ranks.items():
            fused[pos] = fused.get(pos, 0.0) + 1.0 / (RRF_K + rank)

    candidates = []
    for pos, fused_score in fused.items():
        meaning_score = float(np.dot(q_vector[0], index.reconstruct(int(pos))))
        candidates.append((fused_score, meaning_score, pos))
    candidates.sort(reverse=True)
    return candidates


def retrieve(question):
    """
    Pick the chunks to show the model.
    Returns (list of (meaning_score, chunk), best_meaning_score).
    Pass 1: best combined matches, from at most CAP_RECORDS_PER_TYPE different
            records of each kind (a record may contribute several chunks), and
            only chunks whose meaning-score is within MARGIN of the best one.
    Pass 2: if slots are still free, fill them with the next best matches.
    """
    candidates = hybrid_candidates(question)
    best_meaning = max(c[1] for c in candidates)

    picked = []
    records_by_type = {}
    for cand in candidates:
        if len(picked) == TOP_K:
            break
        fused_score, meaning_score, pos = cand
        if meaning_score < best_meaning - MARGIN:
            continue
        kind = chunks[pos]["type"]
        doc_id = chunks[pos]["doc_id"]
        records = records_by_type.setdefault(kind, set())
        if doc_id not in records and len(records) >= CAP_RECORDS_PER_TYPE:
            continue
        records.add(doc_id)
        picked.append(cand)

    for cand in candidates:
        if len(picked) == TOP_K:
            break
        if cand not in picked:
            picked.append(cand)

    picked.sort(reverse=True)
    results = [(meaning_score, chunks[pos]) for _, meaning_score, pos in picked]
    return results, best_meaning


def retrieve_by_type(question, kind, k=2):
    """
    The best k DIFFERENT records of one kind (for example "cwe") for a question.
    Returns a list of (meaning_score, chunk).
    """
    results = []
    seen = set()
    for fused_score, meaning_score, pos in hybrid_candidates(question, kind=kind):
        chunk = chunks[pos]
        if chunk["doc_id"] in seen:
            continue
        seen.add(chunk["doc_id"])
        results.append((meaning_score, chunk))
        if len(results) == k:
            break
    return results