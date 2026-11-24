"""
Scores the code-review mode on the labelled snippets in code_samples/.
Reads code_eval.json and writes code_eval_results_<model>.json.
Ollama must be running. The snippets are only read, never run.
"""
import json
import re
import time

from review_core import CWE_TEXT, MODELS, review_code

RUNS = 3   # set to 2 or 3 to repeat every file (results vary a little between runs)

which = input("Test set: press Enter for the main set (12 files), or type 'extra' for the extra set (8 files): ")
suffix = "_extra" if which.strip().lower() == "extra" else ""
with open(f"code_eval{suffix}.json", "r", encoding="utf-8") as f:
    cases = json.load(f)

choice = input("Model: press Enter for llama3.2:3b, or type 'qwen' for qwen3:4b (slow!): ")
model = MODELS.get(choice.strip().lower(), MODELS[""])

for case in cases:
    for cwe_id in case["expected"]:
        if cwe_id not in CWE_TEXT:
            print(f"Warning: {case['file']} expects {cwe_id}, which is not in your knowledge base.")

print(f"\nReviewing {len(cases)} files x {RUNS} run(s) with {model}...\n")

quiet = lambda *args, **kwargs: None   # we only want the final scores, not step-by-step printing
results = []
for case in cases:
    with open(f"code_samples/{case['file']}", "r", encoding="utf-8") as f:
        code = f.read()

    for run in range(1, RUNS + 1):
        start = time.time()
        phrases, ids, review = review_code(model, code, log=quiet)
        seconds = time.time() - start

        review_text = review or ""
        cited = sorted(set(re.findall(r"CWE-\d+", review_text)))
        expected = set(case["expected"])
        record = {
            "file": case["file"],
            "kind": case["kind"],
            "run": run,
            "expected": case["expected"],
            "step1_phrases": phrases,
            "retrieved_cwes": ids,
            "cited_cwes": cited,
            "findings_reported": review_text.count("Finding:"),
            "retrieval_hit": bool(expected & set(ids)),
            "detection_hit": bool(expected & set(cited)),
            "false_alarm": case["kind"] == "safe" and "Finding:" in review_text,
            "seconds": round(seconds, 1),
            "review": review_text,
        }
        results.append(record)
        print(f"{case['file']:<30} run {run}  retrieved={ids}  cited={cited}  "
              f"findings={record['findings_reported']}  {seconds:.0f}s")

vulnerable = [r for r in results if r["kind"] == "vulnerable"]
safe = [r for r in results if r["kind"] == "safe"]

print("\n" + "=" * 64)
print(f"Code-review results with {model}")
print("=" * 64)
print(f"Right weakness identified (steps 1-2): {sum(r['retrieval_hit'] for r in vulnerable)}/{len(vulnerable)}")
print(f"Right CWE cited in the review:         {sum(r['detection_hit'] for r in vulnerable)}/{len(vulnerable)}")
print(f"False alarms on safe code:             {sum(r['false_alarm'] for r in safe)}/{len(safe)}")
print(f"Average findings per vulnerable file:  {sum(r['findings_reported'] for r in vulnerable) / max(len(vulnerable), 1):.1f}")
print(f"Average time per file:                 {sum(r['seconds'] for r in results) / len(results):.0f} s")
print("=" * 64)

name = f"code_eval_results_{model.split(':')[0]}{suffix}.json"
with open(name, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print(f"Saved {name} (it includes every full review, for your error analysis).")