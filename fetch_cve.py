import requests
import json

# The official NVD API address
url = "https://services.nvd.nist.gov/rest/json/cves/2.0"

# We're asking for CVEs mentioning "SQL injection", limited to 20 results for now
params = {
       "keywordSearch": "SQL injection",
       "resultsPerPage": 20
   }

print("Asking NVD for data... please wait a few seconds.")
response = requests.get(url, params=params)

if response.status_code == 200:
    data = response.json()
    print("Success! Number of CVEs received:", len(data["vulnerabilities"]))

    # Save the raw data to a file so we can look at it and use it later
    with open("cve_data_raw.json", "w") as f:
       json.dump(data, f, indent=2)

       print("Saved to cve_data_raw.json")
else:
   print("Something went wrong. Status code:", response.status_code)