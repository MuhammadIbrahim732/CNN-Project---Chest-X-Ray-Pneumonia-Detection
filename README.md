# 🩻 Chest X-Ray Pneumonia Detection (CNN)

A deep learning project that classifies chest X-ray images as **NORMAL** or **PNEUMONIA** using a Convolutional Neural Network built with TensorFlow/Keras. The project covers the full image classification pipeline: data acquisition from Kaggle, visualization, preprocessing, augmentation, CNN training with class-imbalance handling, and evaluation.

## 📊 Dataset

- **[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)** dataset by Paul Mooney, downloaded via the Kaggle API
- Organized into `train`, `val`, and `test` folders, each split into `NORMAL` and `PNEUMONIA` classes
- Split sizes:

  | Split | Images | Classes |
  |---|---|---|
  | Train | 5,216 | NORMAL, PNEUMONIA |
  | Validation | 16 | NORMAL, PNEUMONIA |
  | Test | 624 | NORMAL, PNEUMONIA |

- The training set is notably **class-imbalanced** toward PNEUMONIA, which the project addresses explicitly (see below)

## 🛠️ Workflow

1. **Import Libraries** — NumPy, Pandas, Matplotlib, Seaborn, PIL, TensorFlow/Keras, Scikit-learn
2. **Load Dataset from Kaggle** — downloaded and extracted via the Kaggle API/CLI
3. **Explore Dataset** — inspect folder structure (`train`/`val`/`test`, each with `NORMAL`/`PNEUMONIA` subfolders)
4. **Visualize Images** — sample NORMAL and PNEUMONIA X-rays displayed for a visual sanity check
5. **Load Dataset into TensorFlow** — `image_dataset_from_directory` for train/val/test, images resized to `150x150`, batch size 32
6. **Visualize the Dataset** — sample batches shown with their class labels
7. **Normalize Images** — pixel values rescaled to `[0, 1]` via `Rescaling(1./255)`
8. **Data Augmentation** — `RandomFlip`, `RandomRotation(0.1)`, `RandomZoom(0.1)` defined (for regularization/generalization)
9. **Build the CNN**:
   ```
   Conv2D(32, 3x3, relu) → MaxPooling2D(2x2)
   Conv2D(64, 3x3, relu) → MaxPooling2D(2x2)
   Flatten
   Dense(128, relu) → Dropout(0.5)
   Dense(1, sigmoid)
   ```
10. **Compile the CNN** — Adam optimizer, binary cross-entropy loss, accuracy metric
11. **Train the CNN** — first trained for 10 epochs without class weighting
12. **Handle Class Imbalance** — computed **balanced class weights** with `sklearn.utils.class_weight`:
    - Class 0 (NORMAL): **1.945**
    - Class 1 (PNEUMONIA): **0.673**
    - Retrained for another 10 epochs using these class weights
13. **Model Evaluation** — evaluated on the held-out test set, plus confusion matrix and classification report
14. **Model Visualization** — training vs. validation accuracy and loss curves plotted across epochs

## 📈 Results

### Test Set Performance (after class-weighted training)

| Metric | Value |
|---|---|
| **Test Accuracy** | **78.2%** |
| Test Loss | 1.846 |

### Confusion Matrix

|  | Predicted NORMAL | Predicted PNEUMONIA |
|---|---|---|
| **Actual NORMAL** | 101 | 133 |
| **Actual PNEUMONIA** | 3 | 387 |

### Classification Report

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| NORMAL | 0.97 | 0.43 | 0.60 | 234 |
| PNEUMONIA | 0.74 | 0.99 | 0.85 | 390 |
| **Accuracy** | | | **0.78** | 624 |
| Macro avg | 0.86 | 0.71 | 0.72 | 624 |
| Weighted avg | 0.83 | 0.78 | 0.76 | 624 |

### Training Progress

- **Without class weights**: training accuracy climbed from 89.4% → ~98%+ over 10 epochs, with validation accuracy fluctuating (62.5%–93.75%), signaling some instability from the tiny 16-image validation set
- **With class weights**: training accuracy reached 99%+ and validation accuracy stabilized near 93.75%–100% by later epochs

> **Key insight:** The model is excellent at catching PNEUMONIA (99% recall) but misses more than half of NORMAL cases (only 43% recall), predicting PNEUMONIA far more often than it should. For a real diagnostic-support tool this is a meaningful limitation — high false positives on NORMAL means many healthy patients would be flagged for pneumonia. Note also that the validation set is extremely small (only 16 images), which limits how much the validation accuracy during training can be trusted — the test set (624 images) is the more reliable signal here.

## 🧰 Tech Stack

- **Python 3**
- **TensorFlow / Keras** — CNN model, image pipeline (`image_dataset_from_directory`), augmentation layers
- **Pandas** & **NumPy** — data handling
- **Matplotlib** & **Seaborn** — visualization
- **PIL (Pillow)** — image loading for exploration
- **Scikit-learn** — class weight computation, confusion matrix, classification report
- **Kaggle API** — dataset download

## 🚀 Getting Started

### Prerequisites

```bash
pip install numpy pandas matplotlib seaborn pillow tensorflow scikit-learn kaggle
```

### Kaggle API Setup

⚠️ **Do not hardcode your Kaggle API key in the notebook.** Instead:

1. Go to [Kaggle Account Settings](https://www.kaggle.com/settings) → "Create New API Token" to download `kaggle.json`
2. Place it at `~/.kaggle/kaggle.json` (Linux/Mac) or use Colab's file upload / `userdata` secrets
3. Set proper permissions: `chmod 600 ~/.kaggle/kaggle.json`

### Data

```bash
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
unzip chest-xray-pneumonia.zip -d chest_xray_data
```

### Run

```bash
jupyter notebook CNN_Project-_Chest_X-Ray_with_Pneumonia_Detection_.ipynb
```

Run all cells sequentially to reproduce data loading, training, and evaluation.

## 📁 Project Structure

```
.
├── CNN_Project-_Chest_X-Ray_with_Pneumonia_Detection_.ipynb   # Main notebook: data pipeline, CNN training & evaluation
└── README.md
```

## 🔮 Future Improvements

- Improve NORMAL-class recall — try threshold tuning, more aggressive augmentation, or a higher weight on the NORMAL class
- Use a larger, stratified validation split instead of the dataset's default 16-image `val` folder
- Apply transfer learning (e.g. ResNet50, DenseNet121, EfficientNet pretrained on ImageNet) — commonly outperforms training a small CNN from scratch on medical imaging
- Add early stopping and learning rate scheduling to reduce overfitting
- Use Grad-CAM to visualize which regions of the X-ray the model focuses on — important for interpretability in a medical context
- Store the Kaggle API key securely (environment variable or secrets manager) rather than in the notebook

## ⚠️ Disclaimer

This project is for educational and research purposes only. It is **not** a validated medical diagnostic tool and should not be used for actual clinical decision-making.

## 👤 Author

**Muhammad Ibrahim**

- 📧 Email: [mibrahim.seng@gmail.com](mailto:mibrahim.seng@gmail.com)
- 💼 LinkedIn: [muhammad-ibrahim-python](https://www.linkedin.com/in/muhammad-ibrahim-python)

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
