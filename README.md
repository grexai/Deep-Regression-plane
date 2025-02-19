# Deep Regression Plane (Python Edition)

This repository contains the Python implementation of the Deep Regression Plane project, adapted from its original MATLAB version.

## Data Preparation

### 1. Download Dataset
The dataset can be downloaded from [Zenodo](<insert-zenodo-link-here>).

### 2. Convert Annotations
Once downloaded, the dataset must be converted to JSON format using the `convert_annotation.py` script.

### 3. Directory Structure
Ensure the dataset is organized in the following structure:

```
/storage01/grexai/datasets/Regplane_data/2022_v1_zeroPadded_split_with_test/
├── trainBalAug_v2_2/
│   ├── images/
│   ├── labels/
│   ├── labels2/  # Generated JSON annotations
├── val/
│   ├── images/
│   ├── labels/
│   ├── labels2/
├── test/
│   ├── images/
│   ├── labels/
│   ├── labels2/
```

### 4. Processing Annotations
Each TIFF label file contains coordinate information, which is extracted and rotated 90° counterclockwise around the point `(5000, 5000)`. The processed coordinates are then saved as JSON files in the `labels2/` directory.

---

## Model Training

The training pipeline is implemented in PyTorch and supports multiple architectures:

- **InceptionV3**  
- **ResNet**

Both models are trained on the preprocessed dataset.

---

## Usage

### 1. Export Data from BIAS Software
The dataset should be exported in `ACC` format.

### 2. Run Predictions
Use `predict_for_BIAS.py` to perform inference:

- Reads the input images
- Generates crops and converts them into batches
- Loads an ensemble model (ResNet and InceptionV3)
- Runs ensemble predictions
- Filters predictions where `radius > 3500`

---

## Acknowledgments

This work builds upon previous research and tools developed for deep regression analysis of biological images.
