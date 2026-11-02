import json

with open("attack_data_raw.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Filter down to just the "attack-pattern" objects — these are the actual techniques
techniques = [obj for obj in data["objects"] if obj.get("type") == "attack-pattern"]

print("Total techniques found:", len(techniques))
print()
print("Here are the first 5, as a sample:")
print()

for technique in techniques[:5]:
    name = technique.get("name", "Unknown")
    description = technique.get("description", "No description available")
    print("Name:", name)
    print("Description:", description[:150], "...")  # just first 150 characters
    print("---")