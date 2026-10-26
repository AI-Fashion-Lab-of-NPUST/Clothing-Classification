import os
import json
import csv
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont

# ============================================================
# PATHS
# ============================================================

MODEL_PATH = r"C:\Users\admin\Desktop\Clothing_Classifier\training_results_v6\best_clothing_model_v2.keras"

CLASS_NAMES_PATH = r"C:\Users\admin\Desktop\Clothing_Classifier\training_results_v6\class_names.json"

INPUT_DIR = r"C:\Users\admin\Desktop\Clothing_Classifier\Final_Test_Data"

OUTPUT_DIR = r"C:\Users\admin\Desktop\Clothing_Classifier\Final_Results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

CSV_PATH = os.path.join(
    OUTPUT_DIR,
    "final_results.csv"
)


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = (224, 224)


# ============================================================
# FONT
# ============================================================

def get_font(size, bold=False):

    paths = [
        r"C:\Windows\Fonts\arialbd.ttf"
        if bold else r"C:\Windows\Fonts\arial.ttf",

        r"C:\Windows\Fonts\segoeuib.ttf"
        if bold else r"C:\Windows\Fonts\segoeui.ttf"
    ]

    for path in paths:

        if os.path.exists(path):
            return ImageFont.truetype(path, size)

    return ImageFont.load_default()


font_title = get_font(34, True)
font_conf = get_font(27, True)
font_file = get_font(15, False)


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 65)
print("Final_Results")
print("=" * 65)

print("\nLoading model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

with open(
    CLASS_NAMES_PATH,
    "r",
    encoding="utf-8"
) as f:

    class_names = json.load(f)

if isinstance(class_names, dict):
    class_names = list(class_names.values())

print("Classes:")

for i, name in enumerate(class_names):
    print(f"  {i}: {name}")

print("\nModel loaded successfully.")


# ============================================================
# FIND IMAGES
# ============================================================

image_files = sorted([
    f
    for f in os.listdir(INPUT_DIR)
    if f.lower().endswith(
        (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    )
])

print(
    f"\nFound {len(image_files)} visible test images."
)

print()


# ============================================================
# CSV
# ============================================================

results = []


# ============================================================
# PROCESS
# ============================================================

for i, filename in enumerate(image_files, 1):

    image_path = os.path.join(
        INPUT_DIR,
        filename
    )

    try:

        # ----------------------------------------------------
        # Load image
        # ----------------------------------------------------

        original = Image.open(
            image_path
        ).convert("RGB")

        # ----------------------------------------------------
        # Resize for model
        # ----------------------------------------------------

        img = original.resize(
            IMAGE_SIZE
        )

        arr = np.array(
            img
        ).astype(
            np.float32
        )

        arr = np.expand_dims(
            arr,
            axis=0
        )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        prediction = model.predict(
            arr,
            verbose=0
        )[0]

        class_index = int(
            np.argmax(prediction)
        )

        confidence = float(
            prediction[class_index]
        )

        predicted_class = class_names[
            class_index
        ]

        # ----------------------------------------------------
        # Top 3 predictions
        # ----------------------------------------------------

        top_indices = np.argsort(
            prediction
        )[::-1][:3]

        top3 = []

        for idx in top_indices:

            top3.append(
                (
                    class_names[idx],
                    float(prediction[idx])
                )
            )

        # ----------------------------------------------------
        # Create canvas
        # ----------------------------------------------------

        width, height = original.size

        banner_height = 125
        footer_height = 75

        canvas = Image.new(
            "RGB",
            (
                width,
                height + banner_height + footer_height
            ),
            "white"
        )

        draw = ImageDraw.Draw(
            canvas
        )

        # ----------------------------------------------------
        # Banner color
        # ----------------------------------------------------

        if confidence >= 0.85:

            banner_color = (25, 135, 84)

        elif confidence >= 0.65:

            banner_color = (230, 150, 30)

        else:

            banner_color = (190, 50, 50)

        draw.rectangle(
            [
                0,
                0,
                width,
                banner_height
            ],
            fill=banner_color
        )

        # ----------------------------------------------------
        # Main prediction
        # ----------------------------------------------------

        title = predicted_class.upper()

        bbox = draw.textbbox(
            (0, 0),
            title,
            font=font_title
        )

        title_width = (
            bbox[2] - bbox[0]
        )

        draw.text(
            (
                (width - title_width) / 2,
                8
            ),
            title,
            fill="white",
            font=font_title
        )

        confidence_text = (
            f"Confidence: "
            f"{confidence * 100:.1f}%"
        )

        bbox = draw.textbbox(
            (0, 0),
            confidence_text,
            font=font_conf
        )

        conf_width = (
            bbox[2] - bbox[0]
        )

        draw.text(
            (
                (width - conf_width) / 2,
                52
            ),
            confidence_text,
            fill="white",
            font=font_conf
        )

        # ----------------------------------------------------
        # Top 3
        # ----------------------------------------------------

        top3_text = (
            " | ".join(
                f"{name}: {prob * 100:.1f}%"
                for name, prob in top3
            )
        )

        bbox = draw.textbbox(
            (0, 0),
            top3_text,
            font=font_file
        )

        top3_width = (
            bbox[2] - bbox[0]
        )

        draw.text(
            (
                (width - top3_width) / 2,
                94
            ),
            top3_text,
            fill="white",
            font=font_file
        )

        # ----------------------------------------------------
        # Original image
        # ----------------------------------------------------

        canvas.paste(
            original,
            (
                0,
                banner_height
            )
        )

        # ----------------------------------------------------
        # Footer
        # ----------------------------------------------------

        footer_y = (
            banner_height +
            height
        )

        draw.rectangle(
            [
                0,
                footer_y,
                width,
                footer_y + footer_height
            ],
            fill=(245, 245, 245)
        )

        bbox = draw.textbbox(
            (0, 0),
            filename,
            font=font_file
        )

        filename_width = (
            bbox[2] - bbox[0]
        )

        draw.text(
            (
                (width - filename_width) / 2,
                footer_y + 25
            ),
            filename,
            fill=(50, 50, 50),
            font=font_file
        )

        # ----------------------------------------------------
        # Save visual result
        # ----------------------------------------------------

        output_path = os.path.join(
            OUTPUT_DIR,
            filename
        )

        canvas.save(
            output_path,
            quality=95
        )

        # ----------------------------------------------------
        # Save CSV result
        # ----------------------------------------------------

        results.append([
            filename,
            predicted_class,
            confidence,
            top3[0][0],
            top3[0][1],
            top3[1][0],
            top3[1][1],
            top3[2][0],
            top3[2][1]
        ])

        print(
            f"[{i:02d}/{len(image_files)}] "
            f"{filename} -> "
            f"{predicted_class} "
            f"({confidence * 100:.1f}%)"
        )

    except Exception as e:

        print(
            f"ERROR: {filename} -> {e}"
        )


# ============================================================
# SAVE CSV
# ============================================================

with open(
    CSV_PATH,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "filename",
        "predicted_class",
        "confidence",
        "top1_class",
        "top1_confidence",
        "top2_class",
        "top2_confidence",
        "top3_class",
        "top3_confidence"
    ])

    writer.writerows(
        results
    )


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 65)
print("VISIBLE TEST COMPLETED")
print("=" * 65)

print(
    f"Images processed: "
    f"{len(results)}"
)

print()
print("Results folder:")
print(OUTPUT_DIR)

print()
print("Predictions CSV:")
print(CSV_PATH)

print()
print("Ground truth was NOT used.")
print("This is a pure visual/generalization test.")

print("=" * 65)