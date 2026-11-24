import json

with open("chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

for i, c in enumerate(chunks):
    if c["doc_id"] == "T1566":
        has = "yes" if "Mitigations:" in c["text"] else "no"
        print(i, "mitigations=" + has, "|", c["text"][:100].replace("\n", " | "))