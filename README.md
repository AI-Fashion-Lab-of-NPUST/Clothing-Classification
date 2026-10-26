# Clothing-Type Classifier

A deep learning component of the CCTV-based clothing analysis system that classifies clothing from extracted torso images.

## Overview

The classifier uses **EfficientNetB0** to predict one of five clothing categories from a torso image. Its predictions can be integrated with the clothing-color and clothing-pattern pipelines to generate a unified clothing description.

## Supported Classes

- Hoodie
- Jackets
- Tshirts
- Shirts
- Tops

## Workflow

```text
Torso Image
     ↓
Image Preprocessing
     ↓
EfficientNetB0
     ↓
Clothing-Type Prediction
     ↓
Unified Analysis Output
```

## Input and Output

**Input:** A cropped torso image extracted from CCTV footage.

**Output:** The predicted clothing category from the five supported classes.

## Integration

This classifier is one of three independent components in the overall system:

- **Clothing-Type Classification:** Identifies the clothing category.
- **Clothing-Color Analysis:** Identifies clothing colors.
- **Clothing-Pattern Classification:** Identifies clothing patterns.

The predictions can be combined to provide a more comprehensive description of clothing in CCTV footage.

## Repository Notes

- Keep the class labels consistent between dataset preparation, training, and inference.
- Store datasets and model weights separately from source code when required by the lab's repository policies.
- Refer to the project's configuration and scripts for implementation-specific training and inference instructions.
