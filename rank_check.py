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
    wanted = input("Which record ID do you want to track (for example CWE-79): ").strip()

    vector = model.encode([query], normalize_embeddings=True)
    vector = np.array(vector, dtype="float32")

    # Ask FAISS to rank EVERY chunk, so we can see exactly where the record lands
    scores, positions = index.search(vector, len(chunks))

    print(f"\nBest score overall: {scores[0][0]:.2f}")
    found = False
    for rank, (score, pos) in enumerate(zip(scores[0], positions[0]), start=1):
        if chunks[pos]["doc_id"] == wanted:
            found = True
            preview = chunks[pos]["text"][:90].replace("\n", " | ")
            print(f"  rank {rank:4d}   score {score:.2f}   {preview}")
    if not found:
        print(f"  {wanted} does not exist in chunks.json")