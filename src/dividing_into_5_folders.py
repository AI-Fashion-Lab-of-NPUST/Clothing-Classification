from pathlib import Path
import pandas as pd
import shutil

# ============================================================
# PATH
# ============================================================
BASE_DIR = Path(r"C:\Users\admin\Desktop\Clothing_Classifier\filtered_5_classes")

IMAGE_DIR = BASE_DIR / "images"
CSV_FILE = BASE_DIR / "labels.csv"

# ============================================================
# TARGET CLASSES
# ============================================================
CLASSES = [
    "Tshirts",
    "Shirts",
    "Hoodie",
    "Jackets",
    "Tops"
]

# ============================================================
# CREATE CLASS FOLDERS
# ============================================================
for class_name in CLASSES:
    (BASE_DIR / class_name).mkdir(parents=True, exist_ok=True)

# ============================================================
# READ CSV
# ============================================================
df = pd.read_csv(CSV_FILE)

moved = 0
missing = 0

# ============================================================
# COPY IMAGES INTO CLASS FOLDERS
# ============================================================
for _, row in df.iterrows():

    filename = str(row["image"]).strip()
    label = str(row["label"]).strip()

    if label not in CLASSES:
        continue

    source = IMAGE_DIR / filename
    destination = BASE_DIR / label / filename

    if source.exists():
        shutil.copy2(source, destination)
        moved += 1
    else:
        print("MISSING:", filename)
        missing += 1

# ============================================================
# SUMMARY
# ============================================================
print("\n===================================")
print("DONE")
print("===================================")

for class_name in CLASSES:
    count = len(list((BASE_DIR / class_name).glob("*")))
    print(f"{class_name:10s}: {count}")

print("-----------------------------------")
print(f"Copied : {moved}")
print(f"Missing: {missing}")
print("===================================")