import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Load everything we saved
index = faiss.read_index("index.faiss")
with open("chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)
model = SentenceTransformer("all-MiniLM-L6-v2")

while True:
    question = input("\nAsk a question (or type 'quit'): ")
    if question.lower() == "quit":
        break

    # Turn the question into meaning numbers, same as we did for the chunks
    q_vector = model.encode([question], normalize_embeddings=True)
    q_vector = np.array(q_vector, dtype="float32")

    # Find the 3 closest chunks
    scores, positions = index.search(q_vector, 3)

    print("\nTop 3 matches:")
    for rank in range(3):
        chunk = chunks[positions[0][rank]]
        score = scores[0][rank]
        print(f"\n#{rank + 1}  [{chunk['source']}]  {chunk['doc_id']}  (similarity: {score:.2f})")
        print(chunk["text"][:300], "...")