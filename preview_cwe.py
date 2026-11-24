import csv

with open("cwe_raw.csv", "r", encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

print("Total entries:", len(rows))
print()
print("Column names:")
for name in rows[0].keys():
    print("  -", name)

print()
# Show one famous weakness: SQL injection is CWE-89
for row in rows:
    if row.get("CWE-ID", "").replace("CWE-", "").strip() == "89":
        print("Name:", row.get("Name"))
        print("Description:", row.get("Description", "")[:300])
        print("Raw mitigations (first 300 characters):")
        print(row.get("Potential Mitigations", "")[:300])
        break