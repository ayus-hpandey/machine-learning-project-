
import os
import glob
import csv

DATASET_PATH = r"KiMoRe"

files = glob.glob(
    os.path.join(DATASET_PATH, "**", "*JointPosition*.csv"),
    recursive=True
)

file_path = files[0]

print("Inspecting:", file_path)

# Read original raw text
with open(file_path, "r", encoding="utf-8-sig") as f:
    lines = [f.readline().strip() for _ in range(5)]

print("\nRAW FILE CONTENT:")
for i, line in enumerate(lines):
    print(f"\nLine {i+1}:")
    print(repr(line[:500]))

# Check possible delimiters
print("\nDELIMITER ANALYSIS:")

with open(file_path, "r", encoding="utf-8-sig") as f:
    first_line = f.readline()

for delimiter in [",", ";", "\t", " "]:
    print(
        repr(delimiter),
        "Count:",
        first_line.count(delimiter)
    )

# Test comma-separated parsing
print("\nCSV READER TEST:")

with open(file_path, "r", encoding="utf-8-sig") as f:
    reader = csv.reader(f, delimiter=",")
    for row in list(reader)[:2]:
        print("Values:", len(row))
        print("First 15:", row[:15])
