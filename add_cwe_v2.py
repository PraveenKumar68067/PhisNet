import csv
import json
import re


def clean_mitigations(raw, limit=3):
    """Turn MITRE's packed mitigation text into a short readable list."""
    items = []
    for part in raw.split("::"):
        if "DESCRIPTION:" in part:
            text = part.split("DESCRIPTION:", 1)[1]
            text = text.split(":EFFECTIVENESS")[0].strip()
            text = re.sub(r"\s*\[REF-\d+\]", "", text)   # remove tags like [REF-1482]
            if text:
                items.append(text)
    if not items and raw.strip():
        items = [raw.strip()[:400]]
    return items[:limit]


def clean_impacts(raw):
    """Pull out the list of possible impacts, without repeats."""
    impacts = []
    for impact in re.findall(r"IMPACT:([^:]+)", raw):
        impact = impact.strip()
        if impact and impact not in impacts:
            impacts.append(impact)
    return impacts


def clean_aliases(raw, limit=5):
    """Pull out nicknames such as 'XSS' or 'SQLi' from MITRE's packed text."""
    names = []
    for term in re.findall(r"TERM:([^:]+)", raw):
        term = term.strip()
        if term and term not in names:
            names.append(term)
    return names[:limit]


# 1. Load the existing knowledge base
with open("documents.json", "r", encoding="utf-8") as f:
    documents = json.load(f)

# Remove any old CWE entries, so running this script twice never creates duplicates
documents = [d for d in documents if d["type"] != "cwe"]
before = len(documents)

# 2. Read the CWE spreadsheet
with open("cwe_raw.csv", "r", encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

# 3. Turn each CWE row into one clean "index card"
for row in rows:
    if row.get("Status", "").strip().lower() == "deprecated":
        continue   # skip outdated entries

    number = row.get("CWE-ID", "").replace("CWE-", "").strip()
    name = row.get("Name", "").strip()
    if not number or not name:
        continue

    description = row.get("Description", "").strip()
    extra = row.get("Extended Description", "").strip()[:800]
    impacts = clean_impacts(row.get("Common Consequences", ""))
    mitigations = clean_mitigations(row.get("Potential Mitigations", ""))
    aliases = clean_aliases(row.get("Alternate Terms", ""))

    # Put the nicknames right into the Name line, so every chunk of this entry carries them
    name_line = f"Name: {name}"
    if aliases:
        name_line += f" (also known as: {', '.join(aliases)})"

    lines = [
        f"CWE ID: CWE-{number}",
        name_line,
        f"Type: {row.get('Weakness Abstraction', 'N/A')}",
        f"Description: {description}",
    ]
    if extra:
        lines.append(f"More detail: {extra}")
    if impacts:
        lines.append(f"Possible impacts: {', '.join(impacts)}")
    if mitigations:
        lines.append("Mitigations: " + " | ".join(mitigations))

    documents.append({
        "id": f"CWE-{number}",
        "source": "MITRE CWE",
        "type": "cwe",
        "severity": None,
        "score": None,
        "published": None,
        "text": "\n".join(lines),
    })

print("Records before adding CWE:", before)
print("CWE entries added:", len(documents) - before)
print("Total records now:", len(documents))

# 4. Save the knowledge base
with open("documents.json", "w", encoding="utf-8") as f:
    json.dump(documents, f, indent=2)
print("Saved documents.json")

# 5. Quick check: show what cross-site scripting looks like now
for d in documents:
    if d["id"] == "CWE-79":
        print()
        print("Check - this is what CWE-79 looks like now:")
        print(d["text"][:500])
        break