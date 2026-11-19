
import os
import glob
from pathlib import Path

DATASET_PATH = r"KiMoRe"

files = glob.glob(
    os.path.join(DATASET_PATH, "**", "*JointPosition*.csv"),
    recursive=True
)

file_path = files[0]

print("FILE:", file_path)
print("FILE SIZE:", os.path.getsize(file_path), "bytes")

# Read raw binary data
with open(file_path, "rb") as f:
    raw = f.read(1000)

print("\nFIRST 200 BYTES:")
print(repr(raw[:200]))

print("\nBYTE ORDER MARK:")
print(raw[:4])

# Try different encodings
for encoding in ["utf-8", "utf-16", "utf-16-le", "latin-1"]:

    try:
        with open(file_path, "r", encoding=encoding) as f:
            content = f.read()

        print("\nENCODING:", encoding)
        print("Total characters:", len(content))
        print("Total lines:", len(content.splitlines()))

        non_empty = [
            line for line in content.splitlines()
            if line.strip()
        ]

        print("Non-empty lines:", len(non_empty))

        print("\nFirst 3 non-empty lines:")

        for line in non_empty[:3]:
            print(repr(line[:300]))

    except Exception as e:
        print(encoding, "FAILED:", e)
