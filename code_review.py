"""
Code Security Review (advisory only - the code you give it is NEVER run).
The logic lives in review_core.py; keep both files in the same folder.
"""
import time

from review_core import AVAILABLE, MAX_CODE_CHARS, MODELS, review_code

print("Code Security Review (advisory only - the code is never run)")
choice = input("Model: press Enter for llama3.2:3b (fast), or type 'qwen' for qwen3:4b (slower): ")
model = MODELS.get(choice.strip().lower(), MODELS[""])
print(f"Using {model}.")

while True:
    path = input("\nPath of the code file to review (or 'quit'): ").strip().strip('"')
    if path.lower() == "quit":
        break

    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            code = f.read()
    except OSError as error:
        print("Could not open that file:", error)
        continue

    if len(code) > MAX_CODE_CHARS:
        print(f"Note: the file is long, so only the first {MAX_CODE_CHARS} characters are reviewed.")

    context = input("What does this code do, or what do you want to know? (Enter to skip): ").strip()
    check = input("Any CWE you want checked, for example CWE-798? (Enter to skip): ").strip()

    start = time.time()
    phrases, ids, review = review_code(model, code, context, check)

    if review is None:
        print(f"\nNone of the {len(AVAILABLE)} listed weakness types was found, and no CWE was requested.")
        print("(The review only covers the types listed in review_core.py, so this is not a guarantee the code is safe.)")
        continue

    print("\n" + "=" * 60)
    print("REVIEW")
    print("=" * 60)
    print(review)
    print("=" * 60)
    if "Finding:" not in review:
        print("Note: the answer does not follow the requested format (the model may have run out of room). Try again, or use Llama.")
    print(f"({time.time() - start:.0f} seconds. AI-generated and advisory: verify before using. The code was never run.)")