import csv

with open("cwe_raw.csv", "r", encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

# 79 = cross-site scripting, 89 = SQL injection, 798 = hard-coded credentials
for wanted in ["79", "89", "798"]:
    found = False
    for row in rows:
        if row.get("CWE-ID", "").replace("CWE-", "").strip() == wanted:
            found = True
            print("CWE-" + wanted)
            print("  Name:", row.get("Name"))
            print("  Alternate Terms:", row.get("Alternate Terms", "")[:200])
            print("  Description:", row.get("Description", "")[:150])
            extended = row.get("Extended Description", "")
            print("  Extended Description starts:", extended[:150])
            break
    if not found:
        print("CWE-" + wanted, "is NOT in this file")
    print()