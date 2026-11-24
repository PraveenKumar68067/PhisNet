import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

index = faiss.read_index("index.faiss")
with open("chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)
model = SentenceTransformer("all-MiniLM-L6-v2")

while True:
    query = input("\nQuery (or 'quit'): ")
    if query.lower() == "quit":
        break

    vector = model.encode([query], normalize_embeddings=True)
    vector = np.array(vector, dtype="float32")
    scores, positions = index.search(vector, 10)   # top 10 this time, not 3

    print()
    for rank, (score, pos) in enumerate(zip(scores[0], positions[0]), start=1):
        chunk = chunks[pos]
        has_mitigations = "yes" if "Mitigations:" in chunk["text"] else "no "
        preview = chunk["text"].replace("\n", " | ")[:80]
        print(f"#{rank:2d}  {score:.2f}  {chunk['doc_id']:<10}  mitigations={has_mitigations}  {preview}")