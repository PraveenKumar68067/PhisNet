import requests
import json

# MITRE publishes the whole ATT&CK dataset (Enterprise matrix) as one file
url = "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json"

print("Downloading MITRE ATT&CK data... this file is large, please wait.")
response = requests.get(url)

if response.status_code == 200:
    data = response.json()

    # Save it locally so we don't need to re-download every time
    with open("attack_data_raw.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print("Saved to attack_data_raw.json")
    print("Total objects in the file:", len(data["objects"]))
else:
    print("Something went wrong. Status code:", response.status_code)