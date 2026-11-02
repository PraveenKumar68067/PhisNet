import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

CHUNK_SIZE = 150    # max words per chunk
OVERLAP = 30        # words repeated between neighboring chunks


def split_into_chunks(text):
    """Cut one long text into overlapping pieces. Short texts stay whole."""
    words = text.split()
    if len(words) <= CHUNK_SIZE:
        return [text]

    header = text.split("\n")[0]   # first line, e.g. "Technique ID: T1190"
    chunks = []
    start = 0
    while start < len(words):
        piece = " ".join(words[start:start + CHUNK_SIZE])
        if start > 0:
            piece = header + "\n" + piece   # remind later chunks what they belong to
        chunks.append(piece)
        start += CHUNK_SIZE - OVERLAP
    return chunks


# 1. Load the clean knowledge base from Step 4
with open("documents.json", "r", encoding="utf-8") as f:
    documents = json.load(f)

# 2. Turn every document into one or more chunks
chunks = []
for doc in documents:
    for piece in split_into_chunks(doc["text"]):
        chunks.append({
            "doc_id": doc["id"],
            "source": doc["source"],
            "type": doc["type"],
            "severity": doc["severity"],
            "text": piece,
        })

print("Documents:", len(documents))
print("Chunks created:", len(chunks))

# 3. Load the embedding model (downloads ~90MB the first time only)
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# 4. Convert every chunk into its "meaning numbers"
print("Creating embeddings... this takes a minute or two.")
texts = [c["text"] for c in chunks]
embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
embeddings = np.array(embeddings, dtype="float32")
print("Embedding shape:", embeddings.shape)

# 5. Put them into a FAISS index for fast searching
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

# 6. Save everything so we never have to rebuild it unless the data changes
faiss.write_index(index, "index.faiss")
with open("chunks.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, indent=2)

print("Done! Saved index.faiss and chunks.json")