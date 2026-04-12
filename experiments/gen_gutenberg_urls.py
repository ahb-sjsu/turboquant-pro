import csv

ids = []
with open("/archive/gutenberg/pg_catalog.csv") as f:
    for r in csv.DictReader(f):
        if r.get("Language", "") == "en" and r.get("Type", "") == "Text":
            bid = r.get("Text#", "").strip()
            if bid and bid.isdigit():
                ids.append(bid)

with open("/archive/gutenberg/urls.txt", "w") as f:
    for i in ids:
        url = "https://www.gutenberg.org/cache/epub/" + i + "/pg" + i + ".txt"
        f.write(url + "\n")
        f.write("  out=texts/" + i + ".txt\n")

print(str(len(ids)) + " URLs generated")
