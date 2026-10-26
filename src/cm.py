import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)

# ============================================================
# PATHS
# ============================================================

GROUND_TRUTH = r"C:\Users\admin\Desktop\Clothing_Classifier\clothing_labels.csv"

PREDICTIONS = r"C:\Users\admin\Desktop\Clothing_Classifier\Final_Results\final_results.csv"

OUTPUT_DIR = r"C:\Users\admin\Desktop\Clothing_Classifier\confusion_matrix_Final"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# EXCLUDED IMAGES
# ============================================================



# ============================================================
# LOAD DATA
# ============================================================

print("=" * 65)
print("CLOTHING CLASSIFIER V6-Final — CONFUSION MATRIX")
print("=" * 65)

gt = pd.read_csv(GROUND_TRUTH)
pred = pd.read_csv(PREDICTIONS)

print("\nGround truth columns:")
print(gt.columns.tolist())

print("\nPrediction columns:")
print(pred.columns.tolist())


# ============================================================
# NORMALIZE COLUMN NAMES
# ============================================================

gt.columns = [str(c).strip() for c in gt.columns]
pred.columns = [str(c).strip() for c in pred.columns]


# ============================================================
# FIND REQUIRED COLUMNS
# ============================================================

def find_column(df, candidates):

    for candidate in candidates:

        for col in df.columns:

            if col.lower() == candidate.lower():
                return col

    return None


gt_filename_col = find_column(
    gt,
    [
        "filename",
        "file",
        "image",
        "image_name",
        "image_filename"
    ]
)

# IMPORTANT:
# ground_truth.csv uses "clothing"
gt_class_col = find_column(
    gt,
    [
        "clothing",
        "ground_truth",
        "true_class",
        "class",
        "label",
        "actual_class"
    ]
)

pred_filename_col = find_column(
    pred,
    [
        "filename",
        "file",
        "image",
        "image_name",
        "image_filename"
    ]
)

pred_class_col = find_column(
    pred,
    [
        "predicted_class",
        "prediction",
        "predicted",
        "class"
    ]
)


# ============================================================
# VALIDATE COLUMNS
# ============================================================

if gt_filename_col is None:
    raise ValueError(
        "Could not find filename column in ground_truth.csv"
    )

if gt_class_col is None:
    raise ValueError(
        f"Could not find ground-truth class column. "
        f"Available columns: {gt.columns.tolist()}"
    )

if pred_filename_col is None:
    raise ValueError(
        "Could not find filename column in predictions CSV."
    )

if pred_class_col is None:
    raise ValueError(
        f"Could not find prediction column. "
        f"Available columns: {pred.columns.tolist()}"
    )


print("\nUsing columns:")

print("Ground truth filename:", gt_filename_col)
print("Ground truth class:", gt_class_col)
print("Prediction filename:", pred_filename_col)
print("Prediction class:", pred_class_col)


# ============================================================
# KEEP ONLY NEEDED COLUMNS
# ============================================================

gt = gt[
    [
        gt_filename_col,
        gt_class_col
    ]
].copy()

pred = pred[
    [
        pred_filename_col,
        pred_class_col
    ]
].copy()


gt.columns = [
    "filename",
    "ground_truth"
]

pred.columns = [
    "filename",
    "prediction"
]


# ============================================================
# CLEAN FILENAMES
# ============================================================

gt["filename"] = (
    gt["filename"]
    .astype(str)
    .str.strip()
    .apply(os.path.basename)
)

pred["filename"] = (
    pred["filename"]
    .astype(str)
    .str.strip()
    .apply(os.path.basename)
)

# ============================================================
# CHECK UNMATCHED FILES
# ============================================================

gt_files = set(gt["filename"])
pred_files = set(pred["filename"])

gt_only = sorted(gt_files - pred_files)
pred_only = sorted(pred_files - gt_files)

print()
print("=" * 65)
print("UNMATCHED FILES")
print("=" * 65)

print(f"\nGround-truth files: {len(gt_files)}")
print(f"Prediction files:   {len(pred_files)}")

print("\nFiles in GROUND TRUTH but NOT in predictions:")
if gt_only:
    for f in gt_only:
        print("  -", f)
else:
    print("  None")

print("\nFiles in PREDICTIONS but NOT in ground truth:")
if pred_only:
    for f in pred_only:
        print("  -", f)
else:
    print("  None")

print("=" * 65)
# ============================================================
# MERGE
# ============================================================

df = pd.merge(
    gt,
    pred,
    on="filename",
    how="inner"
)


# ============================================================
# EXCLUDE POOR IMAGES
# ============================================================




# ============================================================
# NORMALIZE LABELS
# ============================================================

def normalize_label(label):

    label = str(label).strip()

    mapping = {

        "Tshirt": "Tshirts",
        "T-Shirt": "Tshirts",
        "Tshirts": "Tshirts",

        "Shirt": "Shirts",
        "Shirts": "Shirts",

        "Hoodie": "Hoodie",

        "Jacket": "Jackets",
        "Jackets": "Jackets",

        "Top": "Tops",
        "Tops": "Tops"
    }

    return mapping.get(label, label)


df["ground_truth"] = df[
    "ground_truth"
].apply(normalize_label)

df["prediction"] = df[
    "prediction"
].apply(normalize_label)


# ============================================================
# CLASS ORDER
# ============================================================

CLASS_ORDER = [
    "Tshirts",
    "Shirts",
    "Hoodie",
    "Jackets",
    "Tops"
]


# ============================================================
# CHECK UNKNOWN LABELS
# ============================================================

unknown_gt = set(df["ground_truth"]) - set(CLASS_ORDER)
unknown_pred = set(df["prediction"]) - set(CLASS_ORDER)

if unknown_gt:
    print("\nWARNING - Unknown ground-truth labels:")
    print(unknown_gt)

if unknown_pred:
    print("\nWARNING - Unknown prediction labels:")
    print(unknown_pred)


# ============================================================
# ACCURACY
# ============================================================

accuracy = accuracy_score(
    df["ground_truth"],
    df["prediction"]
)

print()
print(f"Accuracy: {accuracy * 100:.2f}%")


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    df["ground_truth"],
    df["prediction"],
    labels=CLASS_ORDER,
    target_names=CLASS_ORDER,
    digits=4,
    zero_division=0
)

print()
print("=" * 65)
print("CLASSIFICATION REPORT")
print("=" * 65)
print(report)


# Save report
with open(
    os.path.join(
        OUTPUT_DIR,
        "classification_report.txt"
    ),
    "w",
    encoding="utf-8"
) as f:

    f.write(
        f"Accuracy: {accuracy:.4f}\n\n"
    )

    f.write(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    df["ground_truth"],
    df["prediction"],
    labels=CLASS_ORDER
)


print()
print("=" * 65)
print("CONFUSION MATRIX")
print("=" * 65)

print(
    pd.DataFrame(
        cm,
        index=CLASS_ORDER,
        columns=CLASS_ORDER
    )
)


# ============================================================
# PLOT
# ============================================================

plt.figure(figsize=(9, 7))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=CLASS_ORDER,
    yticklabels=CLASS_ORDER,
    cmap="Blues"
)

plt.xlabel("Predicted Class")
plt.ylabel("Ground Truth Class")

plt.title(
    "Clothing Classifier V6-Final — Confusion Matrix"
)

plt.tight_layout()

cm_path = os.path.join(
    OUTPUT_DIR,
    "confusion_matrix_v6-Final.png"
)

plt.savefig(
    cm_path,
    dpi=300
)

plt.close()


# ============================================================
# MISCLASSIFICATIONS
# ============================================================

errors = df[
    df["ground_truth"] != df["prediction"]
].copy()

errors_path = os.path.join(
    OUTPUT_DIR,
    "misclassified_images.csv"
)

errors.to_csv(
    errors_path,
    index=False,
    encoding="utf-8"
)


# ============================================================
# CORRECT PREDICTIONS
# ============================================================

correct = df[
    df["ground_truth"] == df["prediction"]
].copy()

correct_path = os.path.join(
    OUTPUT_DIR,
    "correct_predictions.csv"
)

correct.to_csv(
    correct_path,
    index=False,
    encoding="utf-8"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 65)
print("FINAL SUMMARY")
print("=" * 65)

print(f"Images evaluated : {len(df)}")
print(f"Correct          : {len(correct)}")
print(f"Incorrect        : {len(errors)}")
print(f"Accuracy         : {accuracy * 100:.2f}%")

print()
print("Confusion matrix:")
print(cm_path)

print()
print("Misclassified images:")
print(errors_path)

print()
print("Classification report:")
print(
    os.path.join(
        OUTPUT_DIR,
        "classification_report.txt"
    )
)

print("=" * 65)