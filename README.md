# HABNet-Based HAB Detection Pipeline

This repository contains a complete pipeline to train and evaluate a deep learning and machine learning based model for Harmful Algal Bloom (HAB) detection using satellite datacubes.

## Structure
- `Model_Training_Pipeline.ipynb`: Main notebook with data loading, preprocessing, ML and DL training, evaluation, and visualizations.
- `requirements.txt`: Required Python packages.
- `README.md`: Instructions for running the notebook.

## Instructions
1. Upload the dataset folder `data_set_3K` under `/kaggle/input/`.
2. Open and run the `Model_Training_Pipeline.ipynb` notebook sequentially.
3. For full dataset (3000 samples), turn **off GPU** in Kaggle settings to access 16 GB RAM.

## Notes
- The notebook includes HABNet-like architecture, EfficientNet, MobileNet, and VGG experiments.
- Uses augmentation and SHAP interpretability.

## Dataset Format
Each sample should follow this structure:
```
data_set_3K/
├── 0/
│   ├── sample1/
│   │   ├── 1/01.png ... 10.png
│   │   ├── 2/01.png ... 10.png
│   │   ...
├── 1/
│   ├── sample2/
│       ├── ...
```

## Output
- Confusion matrices, validation plots, and classification reports.

## Author
Kruthi
