"""
CyberRAG - Streamlit interface for the cybersecurity RAG assistant.

Run it from your project folder (the one that holds index.faiss, chunks.json,
documents.json, retriever.py and review_core.py):

    streamlit run app.py

Ollama must be running. Nothing leaves this laptop.
"""
import html
import json
import re
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

# ---------------- Page setup (must be the first Streamlit call) ----------------
st.set_page_config(page_title="CyberRAG", page_icon="🛡️", layout="wide")

if tuple(int(p) for p in st.__version__.split(".")[:2]) < (1, 31):
    st.error(f"This app needs Streamlit 1.31 or newer (you have {st.__version__}). Run: pip install --upgrade streamlit")
    st.stop()

# ---------------- Settings ----------------
OLLAMA_URL = "http://localhost:11434"
MODELS = {
    "Llama 3.2 3B  ·  fast": "llama3.2:3b",
    "Qwen3 4B  ·  stricter, slower": "qwen3:4b",
}
NUM_CTX = 3072            # same size as the code-review mode, so models are not reloaded when you switch pages
MIN_SCORE = 0.25          # below this similarity the assistant says it has no relevant information
FOLLOW_UP_WORDS = {"it", "its", "this", "that", "these", "those", "they", "them", "such", "same", "above"}
FEEDBACK_FILE = "feedback_log.jsonl"
SAMPLES_DIR = "code_samples"

# Keep this prompt in sync with rag_chat_v5.py
RAG_SYSTEM = (
    "You are a cybersecurity assistant. Use ONLY the context provided; do not "
    "add facts from outside it. If the context answers only part of the "
    "question, answer that part and say briefly what is missing. If the "
    "context has nothing relevant, reply exactly: I do not have enough "
    "information. Cite only record IDs that appear in the context (the ID in square "
    "brackets before each record), and only ones that support your answer. Never cite an "
    "ID from the earlier questions. Write plain "
    "sentences with no headings, do not show your reasoning, and do not "
    "comment on these instructions. Keep the answer under 6 sentences."
)

# Snapshot of the measured results (edit these numbers if you re-run the evaluation)
RETRIEVAL_RESULTS = pd.DataFrame(
    [
        ["Meaning search only", 0.96, 0.43, 0.84],
        ["Keyword search only", 0.79, 0.32, 0.69],
        ["Hybrid (3 searches)", 0.88, 0.43, 0.75],
        ["Hybrid + balanced (final)", 0.92, 0.47, 0.78],
    ],
    columns=["Strategy", "Hit@4", "Precision@4", "MRR"],
)
ANSWER_RESULTS = pd.DataFrame(
    [
        ["Cited a relevant ID", "13 / 24", "1 / 24"],
        ["Cited IDs that are not in the knowledge base", "0 / 26", "25 / 26"],
        ["Correctly refused out-of-scope questions", "2 / 2", "0 / 2"],
        ["Average faithfulness (manual, 1-5)", "about 4.3", "about 2.2"],
        ["Average relevance (manual, 1-5)", "about 4.7", "about 4.7"],
        ["Average time per answer", "4.0 s", "4.6 s"],
    ],
    columns=["Measure (Llama 3.2 3B, 26 questions)", "With search (RAG)", "Without search"],
)
CODE_RESULTS = pd.DataFrame(
    [
        ["v1  free-form description", "2 / 8", "4 / 4", "-"],
        ["v2  numbered list", "4 / 8", "4 / 4", "3 / 5  (3 / 3 false alarms)"],
        ["v3  names from a fixed list", "7 / 8", "4 / 4", "4 / 5  (2 / 3 false alarms)"],
        ["v4  names + plausibility filter (3 runs)", "20 / 24", "0 / 12", "11 / 15  (0 / 9 false alarms)"],
    ],
    columns=["Code-review version (Llama 3.2 3B)", "Main set: found", "Main set: false alarms", "Extra set: found"],
)

# ---------------- Styling ----------------
CSS = """
<style>
.stApp {
  background:
    radial-gradient(1100px 560px at 8% -8%, #14224a 0%, rgba(11,15,26,0) 60%),
    radial-gradient(900px 480px at 100% 0%, #2a1448 0%, rgba(11,15,26,0) 55%),
    #0b0f1a;
}
#MainMenu, footer { visibility: hidden; }
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0f1730 0%, #0b0f1a 100%);
  border-right: 1px solid rgba(0,212,255,.15);
}
.hero {
  padding: 26px 30px; border-radius: 20px; margin-bottom: 18px;
  background: linear-gradient(135deg, rgba(0,212,255,.13), rgba(139,92,246,.13));
  border: 1px solid rgba(0,212,255,.25); box-shadow: 0 0 44px rgba(0,212,255,.08);
}
.hero h1 { margin: 6px 0; font-size: 2.6rem; letter-spacing: .5px; color: #e6edf3; }
.hero h1 span {
  background: linear-gradient(90deg, #00d4ff, #8b5cf6);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero p { color: #9fb0c8; margin: 0 0 12px 0; font-size: 1.02rem; max-width: 920px; }
.badge-top {
  display: inline-block; font-size: .72rem; letter-spacing: 2px; padding: 4px 10px;
  border-radius: 999px; border: 1px solid rgba(0,212,255,.5); color: #00d4ff;
}
.chip {
  display: inline-block; margin: 4px 8px 0 0; padding: 6px 12px; border-radius: 999px;
  background: rgba(255,255,255,.06); border: 1px solid rgba(255,255,255,.10);
  color: #cfe3ff; font-size: .85rem;
}
.chip b { color: #00d4ff; }
.src-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(290px, 1fr)); gap: 10px; margin-top: 10px; }
.src-card {
  padding: 10px 12px; border-radius: 12px; background: rgba(255,255,255,.045);
  border: 1px solid rgba(255,255,255,.08);
}
.src-top { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.src-badge {
  font-size: .68rem; font-weight: 700; letter-spacing: 1px; padding: 2px 8px;
  border-radius: 999px; border: 1px solid;
}
.src-id { font-weight: 700; color: #e6edf3; }
.sev { font-size: .68rem; font-weight: 700; padding: 2px 8px; border-radius: 999px; border: 1px solid; }
.src-score { margin-left: auto; color: #7f8fa9; font-size: .75rem; }
.src-desc { color: #a9b8d0; font-size: .85rem; margin-top: 6px; line-height: 1.35; }
.pill {
  display: inline-block; margin: 4px 8px 4px 0; padding: 6px 14px; border-radius: 999px;
  font-weight: 600; font-size: .9rem; color: #00d4ff;
  background: rgba(0,212,255,.10); border: 1px solid rgba(0,212,255,.45);
}
.warn-box {
  padding: 10px 14px; border-radius: 12px; margin: 8px 0 14px 0; font-size: .9rem;
  background: rgba(255,212,59,.08); border: 1px solid rgba(255,212,59,.35); color: #ffe69c;
}
.note-box {
  padding: 10px 14px; border-radius: 12px; margin: 8px 0 14px 0; font-size: .9rem;
  background: rgba(139,92,246,.10); border: 1px solid rgba(139,92,246,.35); color: #d0bfff;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

TYPE_STYLE = {"cve": ("CVE", "#ff6b6b"), "cwe": ("CWE", "#4dabf7"), "technique": ("ATT&CK", "#b197fc")}
SEVERITY_COLORS = {"CRITICAL": "#ff3b6b", "HIGH": "#ff8c42", "MEDIUM": "#ffd43b", "LOW": "#69db7c"}


# ---------------- Loading the engine (once) ----------------
@st.cache_resource(show_spinner="Loading the search engine (a few seconds the first time)...")
def load_engine():
    import retriever
    import review_core

    seen = {(c["type"], c["doc_id"]) for c in retriever.chunks}
    stats = {
        "records": len(seen),
        "chunks": len(retriever.chunks),
        "cve": sum(1 for t, _ in seen if t == "cve"),
        "cwe": sum(1 for t, _ in seen if t == "cwe"),
        "technique": sum(1 for t, _ in seen if t == "technique"),
    }
    return retriever, review_core, stats


try:
    retriever, review_core, STATS = load_engine()
except Exception as error:   # missing files, wrong folder, missing libraries
    st.error(f"Could not load the search engine: {error}")
    st.info("Run this app from your project folder (the one with index.faiss, chunks.json, documents.json, retriever.py and review_core.py).")
    st.stop()


# ---------------- Ollama helpers ----------------
def ollama_models():
    """Names of installed models, or None if Ollama is not reachable."""
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
        return [m["name"] for m in response.json().get("models", [])]
    except Exception:
        return None


def ensure_only(model):
    """Unload any other model so two models never compete for the 4GB of GPU memory."""
    try:
        running = requests.get(f"{OLLAMA_URL}/api/ps", timeout=3).json().get("models", [])
        for item in running:
            name = item.get("name") or item.get("model")
            if name and name != model:
                requests.post(f"{OLLAMA_URL}/api/generate", json={"model": name, "keep_alive": 0}, timeout=10)
    except Exception:
        pass


def stream_answer(model, system_prompt, user_prompt):
    """Yield the answer piece by piece. Qwen's reasoning (before </think>) is held back."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": True,
        "keep_alive": "30m",
        "options": {"num_ctx": NUM_CTX, "temperature": 0.2, "num_predict": 500},
    }
    hold = model.startswith("qwen3")
    if hold:
        payload["think"] = False
    released = not hold
    buffer = ""
    with requests.post(f"{OLLAMA_URL}/api/chat", json=payload, stream=True, timeout=(10, 300)) as response:
        for line in response.iter_lines():
            if not line:
                continue
            piece = json.loads(line)
            if "error" in piece:
                raise RuntimeError(piece["error"])
            text = piece.get("message", {}).get("content", "")
            if released:
                if text:
                    yield text
            else:
                buffer += text
                if "</think>" in buffer:
                    released = True
                    after = buffer.split("</think>")[-1].lstrip()
                    if after:
                        yield after
            if piece.get("done"):
                break
    if not released and buffer:
        yield buffer.strip()


# ---------------- Display helpers ----------------
def describe(chunk):
    """A short human-readable description of a record, for the source cards."""
    text = chunk["text"]
    match = re.search(r"Name:\s*(.*?)(?=\s+(?:Tactics:|Type:|Description:)|\(also known as|\n|$)", text)
    if match and match.group(1).strip():
        return match.group(1).strip()
    match = re.search(r"Description:\s*(.*)", text)
    return match.group(1).strip()[:140] if match else ""


def source_cards(retrieved):
    """HTML cards for the distinct records that were given to the model."""
    seen, cards = set(), []
    for score, chunk in retrieved:
        if chunk["doc_id"] in seen:
            continue
        seen.add(chunk["doc_id"])
        label, color = TYPE_STYLE.get(chunk["type"], ("SRC", "#adb5bd"))
        severity = chunk.get("severity")
        severity_html = ""
        if severity in SEVERITY_COLORS:
            sev_color = SEVERITY_COLORS[severity]
            severity_html = (
                f'<span class="sev" style="background:{sev_color}22;color:{sev_color};'
                f'border-color:{sev_color}">{severity}</span>'
            )
        cards.append(
            f'<div class="src-card" style="border-left:4px solid {color}">'
            f'<div class="src-top"><span class="src-badge" style="color:{color};border-color:{color}">{label}</span>'
            f'<span class="src-id">{html.escape(chunk["doc_id"])}</span>{severity_html}'
            f'<span class="src-score">similarity {score:.2f}</span></div>'
            f'<div class="src-desc">{html.escape(describe(chunk))}</div></div>'
        )
    return '<div class="src-grid">' + "".join(cards) + "</div>"


def build_prompts(question, retrieved, history_block):
    context = "".join(f"[{c['source']} - {c['doc_id']}]\n{c['text']}\n\n" for _, c in retrieved)
    return RAG_SYSTEM, f"{history_block}Context:\n{context}\nQuestion: {question}"


def history_pairs(n=2):
    pairs = [(h["question"], h["content"]) for h in st.session_state.history if h["role"] == "assistant"]
    return pairs[-n:]


def history_block(pairs):
    if not pairs:
        return ""
    lines = ["Earlier questions in this chat (only to understand what words like 'it' refer to; never use them as facts):"]
    for q, _answer in pairs:
        lines.append(f"- {q}")
    return "\n".join(lines) + "\n\n"


def save_feedback(item, rating):
    record = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "rating": rating,
        "question": item["question"],
        "answer": item["content"],
        "model": item["model"],
        "sources": item["source_ids"],
    }
    with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def feedback_count():
    path = Path(FEEDBACK_FILE)
    return sum(1 for _ in open(path, encoding="utf-8")) if path.exists() else 0


# ---------------- Session state ----------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- Sidebar ----------------
with st.sidebar:
    st.markdown("### 🛡️ CyberRAG")
    page = st.radio(
        "Navigate",
        ["🔎 Ask", "🧪 Code review", "📈 Evaluation", "🧭 How it works"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("##### ⚙️ Settings")
    model_label = st.selectbox("Language model", list(MODELS))
    model = MODELS[model_label]
    use_memory = st.toggle("Remember the last 2 turns", value=True)

    installed = ollama_models()
    if installed is None:
        st.error("Ollama is not running. Start it from the Start menu, then refresh this page.")
    else:
        st.success("Ollama is online")
        if model not in installed:
            st.warning(f"{model} is not installed. Run: ollama pull {model}")

    if st.button("🧹 Clear chat"):
        st.session_state.history = []
        st.rerun()
    st.caption(f"Feedback saved so far: {feedback_count()}")
    st.divider()
    st.caption(
        f"Search: top {retriever.TOP_K} chunks, at most {retriever.CAP_RECORDS_PER_TYPE} records per source type. "
        "Everything runs locally; no data leaves this laptop."
    )

# ---------------- Header ----------------
st.markdown(
    f"""
<div class="hero">
  <span class="badge-top">LOCAL · PRIVATE · SOURCE-CITED</span>
  <h1>Cyber<span>RAG</span></h1>
  <p>A threat-intelligence assistant over NVD vulnerabilities, MITRE ATT&amp;CK techniques and MITRE CWE weaknesses.
  Answers cite the records they came from, and everything runs on this laptop.</p>
  <span class="chip"><b>{STATS['records']:,}</b> records</span>
  <span class="chip"><b>{STATS['chunks']:,}</b> searchable chunks</span>
  <span class="chip"><b>{STATS['technique']:,}</b> ATT&amp;CK techniques</span>
  <span class="chip"><b>{STATS['cwe']:,}</b> CWE weaknesses</span>
  <span class="chip"><b>{STATS['cve']:,}</b> CVEs</span>
  <span class="chip">hybrid search · FAISS + BM25</span>
</div>
""",
    unsafe_allow_html=True,
)


# =====================================================================
# PAGE 1: ASK
# =====================================================================
def render_history():
    for i, item in enumerate(st.session_state.history):
        avatar = "🧑‍💻" if item["role"] == "user" else "🛡️"
        with st.chat_message(item["role"], avatar=avatar):
            st.markdown(item["content"])
            if item["role"] == "assistant":
                if item.get("sources_html"):
                    st.markdown(item["sources_html"], unsafe_allow_html=True)
                st.caption(item["meta"])
                up, down, _ = st.columns([1, 1, 10])
                if up.button("👍", key=f"up{i}"):
                    save_feedback(item, "up")
                    st.toast("Thanks for the feedback!")
                if down.button("👎", key=f"down{i}"):
                    save_feedback(item, "down")
                    st.toast("Thanks, noted.")


def handle_question(question):
    pairs = history_pairs() if use_memory else []
    query = question
    words = re.findall(r"[a-z']+", question.lower())
    refers_back = any(word in FOLLOW_UP_WORDS for word in words)
    if pairs and (refers_back or len(words) < 8):
        # a follow-up: search with the previous search query plus this question
        previous = st.session_state.get("last_query") or pairs[-1][0]
        query = (previous + " " + question)[-300:]
    st.session_state.last_query = query

    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="🛡️"):
        started = time.time()
        retrieved, best_score = retriever.retrieve(query)
        search_ms = (time.time() - started) * 1000
        sources_html = ""
        generate_seconds = 0.0

        if best_score < MIN_SCORE:
            answer = "I could not find relevant information in my knowledge base."
            st.markdown(answer)
        else:
            try:
                with st.spinner("Preparing the model..."):
                    ensure_only(model)
                system_prompt, user_prompt = build_prompts(question, retrieved, history_block(pairs))
                t1 = time.time()
                answer = st.write_stream(stream_answer(model, system_prompt, user_prompt))
                generate_seconds = time.time() - t1
            except Exception as error:
                st.error(f"Could not get an answer from Ollama: {error}")
                return
            sources_html = source_cards(retrieved)
            st.markdown(sources_html, unsafe_allow_html=True)

        meta = f"🧠 {model} · search {search_ms:.0f} ms · answer {generate_seconds:.1f} s"
        st.caption(meta)

    st.session_state.history.append({"role": "user", "content": question})
    st.session_state.history.append(
        {
            "role": "assistant",
            "content": answer,
            "sources_html": sources_html,
            "meta": meta,
            "question": question,
            "model": model,
            "source_ids": sorted({c["doc_id"] for _, c in retrieved}),
        }
    )
    st.rerun()


if page == "🔎 Ask":
    if not st.session_state.history:
        st.markdown("##### Try an example")
        examples = [
            "What is phishing and how do adversaries use it?",
            "How does SQL injection work and how can it be prevented?",
            "What is cross-site scripting?",
            "Which SQL injection flaw affects Oracle IAS 3.0.7?",
        ]
        columns = st.columns(len(examples))
        for column, example in zip(columns, examples):
            if column.button(example, key=f"example_{example}"):
                st.session_state.pending = example

    render_history()

    question = st.chat_input("Ask about a CVE, an attack technique or a weakness...")
    if not question:
        question = st.session_state.pop("pending", None)
    if question:
        handle_question(question)

# =====================================================================
# PAGE 2: CODE REVIEW
# =====================================================================
elif page == "🧪 Code review":
    st.markdown("### 🧪 Code security review")
    st.markdown(
        '<div class="warn-box">Advisory only. The code is <b>never run</b>. The review is AI-generated, covers just '
        f"{len(review_core.AVAILABLE)} weakness types, and its suggested fixes can be wrong: verify before using.</div>",
        unsafe_allow_html=True,
    )

    samples = sorted(p.name for p in Path(SAMPLES_DIR).glob("*.py")) if Path(SAMPLES_DIR).exists() else []
    for extra in ("sample_hardcoded_password.py", "sample_sql_injection.py"):
        if Path(extra).exists():
            samples.append(extra)

    def load_sample():
        choice = st.session_state.sample_choice
        if choice != "-":
            path = Path(SAMPLES_DIR) / choice if (Path(SAMPLES_DIR) / choice).exists() else Path(choice)
            st.session_state.code_text = path.read_text(encoding="utf-8", errors="replace")

    left, right = st.columns([3, 2])
    with left:
        if samples:
            st.selectbox("Load a sample file", ["-"] + samples, key="sample_choice", on_change=load_sample)
        uploaded = st.file_uploader("...or upload a source file", type=["py", "txt", "c", "cpp", "js", "java", "php"])
        st.text_area("Code to review", key="code_text", height=300, placeholder="Paste code here...")
    with right:
        context = st.text_area(
            "What does this code do, or what do you want to know? (optional)", height=110, key="review_context"
        )
        check = st.text_input("Any CWE you want checked? (optional, for example CWE-798)", key="review_check")
        run = st.button("🔍 Run security review", type="primary")

    code = uploaded.getvalue().decode("utf-8", errors="replace") if uploaded is not None else st.session_state.get("code_text", "")

    if run:
        if not code.strip():
            st.warning("Paste some code, load a sample, or upload a file first.")
        else:
            if len(code) > review_core.MAX_CODE_CHARS:
                st.info(f"The code is long, so only the first {review_core.MAX_CODE_CHARS} characters are reviewed.")
            started = time.time()
            try:
                with st.status("Reviewing the code...", expanded=True) as status:
                    ensure_only(model)
                    labels, ids, review = review_core.review_code(
                        model, code, context, check, log=lambda message: st.write(message.strip())
                    )
                    status.update(label=f"Review finished in {time.time() - started:.0f} s", state="complete")
            except Exception as error:
                st.error(f"The review failed: {error}")
                st.stop()

            st.markdown("#### Weakness types found")
            if labels:
                st.markdown("".join(f'<span class="pill">{html.escape(label)}</span>' for label in labels), unsafe_allow_html=True)
            else:
                if review_core.looks_like_c(code):
                    st.warning(
                        "None of the listed weakness types was found, but this looks like C/C++ code, which this review "
                        "covers only partly. Also use g++ -Wall -Wextra -fsanitize=address or cppcheck."
                    )
                else:
                    st.success("None of the listed weakness types was found.")
                    st.caption("The review only covers the types in review_core.py, so this is not a guarantee that the code is safe.")

            if review:
                st.markdown("#### Review")
                with st.container(border=True):
                    st.markdown(review)
                st.markdown("#### Reference entries from the knowledge base")
                for cwe_id in ids:
                    with st.expander(f"{cwe_id}  ·  {review_core.short_name(review_core.CWE_TEXT[cwe_id])}"):
                        st.text(review_core.compact_cwe(review_core.CWE_TEXT[cwe_id]))

# =====================================================================
# PAGE 3: EVALUATION
# =====================================================================
elif page == "📈 Evaluation":
    st.markdown("### 📈 How well does it work?")
    st.markdown(
        '<div class="note-box">A snapshot of the results measured on this laptop, using a small hand-made test set '
        "(24 retrieval questions, 26 answer questions, 20 code files). Treat the numbers as indicative, not final.</div>",
        unsafe_allow_html=True,
    )

    st.markdown("#### 1. Does the search find the right records?")
    st.dataframe(RETRIEVAL_RESULTS, hide_index=True)
    st.bar_chart(RETRIEVAL_RESULTS.set_index("Strategy")[["Hit@4", "Precision@4", "MRR"]])
    st.caption(
        "Plain meaning search scored as well as the hybrid on this set (a difference of about one question). "
        "The hybrid was built to fix specific failures such as XSS being confused with CSRF."
    )

    st.markdown("#### 2. Are the answers grounded? (with search vs the same model without search)")
    st.dataframe(ANSWER_RESULTS, hide_index=True)
    st.caption("Without search, IDs not found in the knowledge base are not necessarily invented; many may be real but unrelated CVEs.")

    st.markdown("#### 3. Does the code review catch real problems?")
    st.dataframe(CODE_RESULTS, hide_index=True)
    st.caption(
        "v4 adds a plausibility filter whose rules were written after seeing the first results, so its low false-alarm "
        "rate on these files is partly by design. Path traversal and disabled-TLS checks were missed in every run."
    )

# =====================================================================
# PAGE 4: HOW IT WORKS
# =====================================================================
else:
    st.markdown("### 🧭 How it works")
    st.graphviz_chart(
        """
digraph G {
  rankdir=LR; bgcolor="transparent";
  node [shape=box, style="rounded,filled", fillcolor="#131a2b", color="#00d4ff", fontcolor="#e6edf3", fontname="Helvetica"];
  edge [color="#6b7a99"];
  Q [label="Question"];
  M [label="Meaning search\\n(embeddings + FAISS)"];
  K [label="Keyword search\\n(BM25)"];
  T [label="Name search\\n(BM25 on titles)"];
  F [label="Rank fusion +\\nbalanced selection"];
  L [label="Local LLM\\n(Ollama)"];
  A [label="Answer + source cards"];
  Q -> M; Q -> K; Q -> T;
  M -> F; K -> F; T -> F;
  F -> L -> A;
}
"""
    )
    left, right = st.columns(2)
    with left:
        st.markdown(
            """
**Knowledge base**
- NVD vulnerabilities (CVEs) with severity scores
- MITRE ATT&CK techniques with mitigations
- MITRE CWE weakness types with nicknames and mitigations

**Ask**
- Three searches are combined by reciprocal rank fusion
- No more than a few records per source type, so CVEs cannot crowd out explanations
- The model must answer from the retrieved records and cite them; below a similarity threshold it refuses
"""
        )
    with right:
        st.markdown(
            """
**Code review**
1. Clear code patterns (strcpy, eval, verify=False, ...) are matched directly by rules
2. The model picks weakness types from a fixed list
3. A plausibility filter drops model picks with no matching code pattern
4. The matching CWE entries are read from the knowledge base
5. The model explains the problem and suggests a fix

**Limits**
- Only the listed weakness types are checked
- Suggested fixes are unverified and sometimes wrong
- Small local models are slower and less reliable than hosted ones
"""
        )