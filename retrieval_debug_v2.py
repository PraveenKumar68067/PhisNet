from retriever import hybrid_candidates, chunks

while True:
    query = input("\nQuery (or 'quit'): ")
    if query.lower() == "quit":
        break

    candidates = hybrid_candidates(query)

    print()
    print("rank  fused    meaning  record        start of chunk")
    for rank, (fused_score, meaning_score, pos) in enumerate(candidates[:10], start=1):
        chunk = chunks[pos]
        preview = chunk["text"].replace("\n", " | ")[:70]
        print(f"#{rank:2d}   {fused_score:.4f}   {meaning_score:.2f}     {chunk['doc_id']:<12}  {preview}")