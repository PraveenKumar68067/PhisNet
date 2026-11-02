import json

documents = []   # an empty list; we'll add one clean record per CVE/technique


# ---------- PART 1: Clean the CVE data ----------

with open("cve_data_raw.json", "r", encoding="utf-8") as f:
    cve_data = json.load(f)

for item in cve_data["vulnerabilities"]:
    cve = item["cve"]
    cve_id = cve["id"]
    published = cve.get("published", "unknown")[:10]   # keep just the date part

    # Find the English description
    description = ""
    for d in cve["descriptions"]:
        if d["lang"] == "en":
            description = d["value"]
            break

    # Find the severity score (newer scoring versions first, older as backup)
    score = "N/A"
    severity = "N/A"
    metrics = cve.get("metrics", {})
    for key in ["cvssMetricV31", "cvssMetricV30", "cvssMetricV2"]:
        if key in metrics:
            metric = metrics[key][0]
            score = metric["cvssData"]["baseScore"]
            severity = metric["cvssData"].get("baseSeverity") or metric.get("baseSeverity", "N/A")
            break

    text = (
        f"CVE ID: {cve_id}\n"
        f"Severity: {severity} (score {score})\n"
        f"Published: {published}\n"
        f"Description: {description}"
    )

    documents.append({
        "id": cve_id,
        "source": "NVD",
        "type": "cve",
        "severity": severity,
        "score": score,
        "published": published,
        "text": text,
    })

print("CVE records cleaned:", len(documents))


# ---------- PART 2: Clean the ATT&CK data ----------

with open("attack_data_raw.json", "r", encoding="utf-8") as f:
    attack_data = json.load(f)

objects = attack_data["objects"]

# First, collect all "mitigations" (defenses) so we can attach them to techniques
defenses = {}
for obj in objects:
    if obj.get("type") == "course-of-action":
        defenses[obj["id"]] = obj

# Next, find which defense protects against which technique
defenses_for_technique = {}
for obj in objects:
    if obj.get("type") == "relationship" and obj.get("relationship_type") == "mitigates":
        defense = defenses.get(obj["source_ref"])
        if defense:
            defenses_for_technique.setdefault(obj["target_ref"], []).append(defense["name"])

count_before = len(documents)

for obj in objects:
    if obj.get("type") != "attack-pattern":
        continue
    if obj.get("revoked") or obj.get("x_mitre_deprecated"):
        continue   # skip outdated techniques

    # Find the official technique ID (like T1190)
    technique_id = ""
    for ref in obj.get("external_references", []):
        if ref.get("source_name") == "mitre-attack":
            technique_id = ref.get("external_id", "")
            break

    name = obj.get("name", "Unknown")
    description = obj.get("description", "")
    tactics = [p["phase_name"] for p in obj.get("kill_chain_phases", [])]
    defenses_list = defenses_for_technique.get(obj["id"], [])

    text = (
        f"Technique ID: {technique_id}\n"
        f"Name: {name}\n"
        f"Tactics: {', '.join(tactics) if tactics else 'N/A'}\n"
        f"Description: {description}\n"
        f"Mitigations: {', '.join(defenses_list) if defenses_list else 'None listed'}"
    )

    documents.append({
        "id": technique_id,
        "source": "MITRE ATT&CK",
        "type": "technique",
        "severity": None,
        "score": None,
        "published": None,
        "text": text,
    })

print("ATT&CK techniques cleaned:", len(documents) - count_before)


# ---------- PART 3: Save everything in one clean file ----------

with open("documents.json", "w", encoding="utf-8") as f:
    json.dump(documents, f, indent=2)

print("Total clean records saved to documents.json:", len(documents))