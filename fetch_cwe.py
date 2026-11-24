import io
import zipfile
import requests

# MITRE's full list of software weaknesses, as a zipped spreadsheet (CSV)
url = "https://cwe.mitre.org/data/csv/1000.csv.zip"

print("Downloading CWE data...")
response = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (student project)"}, timeout=60)

if response.status_code == 200:
    # The download is a .zip file. Open it from memory and take out the CSV inside.
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        inside_name = z.namelist()[0]
        with open("cwe_raw.csv", "wb") as out:
            out.write(z.read(inside_name))
    print("Saved to cwe_raw.csv")
else:
    print("Something went wrong. Status code:", response.status_code)