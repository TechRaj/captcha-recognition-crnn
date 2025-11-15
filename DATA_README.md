# Dataset Information

This project uses a CAPTCHA dataset that is **not included** in this repository due to size constraints.

## Dataset Details

- **Training images**: 8,010 images (7,777 after manual pruning)
- **Test images**: 2,000 images
- **Total size**: ~60MB uncompressed
- **Format**: PNG images with labels in filename (e.g., `label-0.png`)

## Download Dataset

📦 **Kaggle Dataset**: https://www.kaggle.com/datasets/techraj/captcha-training-images

## Setup Instructions

After downloading the dataset from Kaggle:

1. Extract the dataset files to the project root:
   ```
   captacha_dataset/
   ├── train/     (8,010 images)
   └── test/      (2,000 images)
   ```

2. Verify the folder structure matches the expected layout
3. Run the notebook `mini-proj.ipynb`

## Alternative: Use Kaggle API

```bash
# Install Kaggle CLI
pip install kaggle

# Download dataset
kaggle datasets download -d techraj/captcha-training-images

# Extract
unzip captcha-training-images.zip
```

## Referenced Papers

The `referenced_papers/` folder contains research papers used for this project.
See the bibliography section in `model_logs.md` for full citations.
