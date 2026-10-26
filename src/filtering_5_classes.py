from pathlib import Path
import pandas as pd
import shutil

# ============================================================
# CHANGE THIS TO YOUR LAMBI FOLDER
# ============================================================
BASE_DIR = Path(r"C:\Users\admin\Desktop\Clothing_Classifier\LAMBI")

# Original files
IMAGE_DIR = BASE_DIR / "1000img_class"
CSV_FILE = BASE_DIR / "1000img.csv"

# New clean dataset
OUTPUT_DIR = BASE_DIR / "filtered_5_classes"
OUTPUT_IMAGE_DIR = OUTPUT_DIR / "images"
OUTPUT_CSV = OUTPUT_DIR / "labels.csv"

# ============================================================
# TARGET CLASSES
# ============================================================
TARGET_CLASSES = {
    "Tshirts",
    "Shirts",
    "Hoodie",
    "Jackets",
    "Tops"
}

# ============================================================
# CREATE OUTPUT FOLDERS
# ============================================================
OUTPUT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# READ CSV
# ============================================================
df = pd.read_csv(CSV_FILE)

print("CSV columns:")
print(df.columns.tolist())

print("\nOriginal class distribution:")
print(df["label"].value_counts())

# ============================================================
# KEEP ONLY OUR 5 CLASSES
# ============================================================
filtered_df = df[df["label"].isin(TARGET_CLASSES)].copy()

print("\nFiltered class distribution:")
print(filtered_df["label"].value_counts())

# ============================================================
# COPY ONLY THE 5-CLASS IMAGES
# ============================================================
copied = 0
missing = 0

for _, row in filtered_df.iterrows():

    filename = str(row["image"]).strip()

    source = IMAGE_DIR / filename
    destination = OUTPUT_IMAGE_DIR / filename

    if source.exists():
        shutil.copy2(source, destination)
        copied += 1
    else:
        print("MISSING:", filename)
        missing += 1

# ============================================================
# SAVE CLEAN CSV
# ============================================================
# Keep only image + label
clean_df = filtered_df[["image", "label"]].copy()

clean_df.to_csv(
    OUTPUT_CSV,
    index=False,
    encoding="utf-8-sig"
)

# ============================================================
# SUMMARY
# ============================================================
print("\n========================================")
print("DONE")
print("========================================")

print(f"Images copied : {copied}")
print(f"Images missing: {missing}")
print(f"Total records : {len(clean_df)}")

print("\nClasses:")
print(clean_df["label"].value_counts())

print("\nOutput:")
print(OUTPUT_DIR)