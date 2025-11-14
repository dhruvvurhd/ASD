# ASD (Autism Spectrum Disorder) Prediction Model

A machine learning model for predicting Autism Spectrum Disorder (ASD) based on screening questionnaire responses and demographic information.

## 📋 Overview

This project implements a binary classification model that predicts ASD diagnosis using:
- 10 screening questionnaire responses (A1-A10)
- Demographic information (Age, Sex)
- Medical history (Jauundice, Family_ASD)

**Model Type**: Supervised Learning - Binary Classification  
**Best Model**: Gradient Boosting Classifier  
**Accuracy**: 99.84% (Test Set)

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/dhruvvurhd/ASD.git
cd ASD

# Install dependencies
pip install -r requirements.txt
```

### Training the Model

```bash
python asd_prediction_model.py
```

This will:
- Load and preprocess the dataset
- Train 4 different ML models (Random Forest, Gradient Boosting, SVM, Logistic Regression)
- Select the best model based on performance
- Save the trained model as `asd_model.pkl`

### Making Predictions

#### Option 1: Python Script
```python
from predict_asd import predict_asd

# Single prediction
data = {
    'A1': 1, 'A2': 1, 'A3': 1, 'A4': 1, 'A5': 1,
    'A6': 1, 'A7': 1, 'A8': 1, 'A9': 1, 'A10': 1,
    'Age': 15,
    'Sex': 'm',
    'Jauundice': 'no',
    'Family_ASD': 'yes'
}

results = predict_asd(data)
print(f"Prediction: {results[0]['ASD_Prediction']}")
print(f"Probability: {results[0]['Probability']}")
```

#### Option 2: Command Line
```bash
python predict_asd.py
```

#### Option 3: Batch Predictions
```bash
python predict_asd.py your_data.csv
```

## 📊 Dataset

**File**: `Autism_Screening_Data_Combined.csv`

- **Samples**: 6,075
- **Features**: 14 (A1-A10, Age, Sex, Jauundice, Family_ASD)
- **Target**: Class (YES/NO)

**Class Distribution**:
- NO (No ASD): 4,271 (70.3%)
- YES (Has ASD): 1,804 (29.7%)

## 🏗️ Project Structure

```
ASD/
├── asd_prediction_model.py    # Main training script
├── predict_asd.py              # Prediction interface
├── requirements.txt            # Python dependencies
├── Autism_Screening_Data_Combined.csv  # Dataset
├── asd_model.pkl              # Trained model (generated)
├── README.md                   # This file
└── RESULTS.md                  # Detailed results and analysis
```

## 📈 Model Performance

### Best Model: Gradient Boosting

| Metric | Score |
|--------|-------|
| **Test Accuracy** | 99.84% |
| **ROC-AUC** | 0.9999 |
| **Cross-Validation Accuracy** | 99.24% (±0.74%) |

### Model Comparison

| Model | Accuracy | ROC-AUC | CV Accuracy |
|-------|----------|---------|-------------|
| Gradient Boosting | 99.84% | 0.9999 | 99.24% |
| Random Forest | 98.11% | 0.9989 | 98.02% |
| SVM | 96.05% | 0.9922 | 97.00% |
| Logistic Regression | 89.71% | 0.9575 | 90.62% |

### Feature Importance

Top 5 Most Important Features:
1. **A6** (Question 6): 23.02%
2. **Age**: 22.95%
3. **A9** (Question 9): 10.87%
4. **A4**: 8.13%
5. **A5**: 7.95%

See [RESULTS.md](RESULTS.md) for detailed analysis.

## 🔧 Technical Details

### Algorithms Tested
- **Random Forest**: Ensemble of decision trees
- **Gradient Boosting**: Sequential ensemble learning
- **SVM**: Support Vector Machine with RBF kernel
- **Logistic Regression**: Linear probabilistic model

### Preprocessing
- Categorical encoding (Label Encoding)
- Feature scaling (StandardScaler for SVM/LR)
- Missing value handling
- Train/Test split: 80/20 (stratified)

### Model Selection
Models are evaluated on:
- Test Accuracy
- ROC-AUC Score
- 5-fold Cross-Validation

The model with the highest ROC-AUC is selected.

## ⚠️ Important Notes

### Limitations
- **High Accuracy**: The 99.84% accuracy may be due to dataset characteristics (strong feature separation)
- **Dataset**: May not represent real-world clinical complexity
- **Generalization**: Model may not generalize to different populations or screening tools

### Clinical Disclaimer
⚠️ **This model is for research/educational purposes only**
- Not a medical device
- Should not be used for actual diagnosis
- Does not replace professional medical evaluation

## 📝 Requirements

- Python 3.7+
- pandas >= 1.3.0
- numpy >= 1.21.0
- scikit-learn >= 1.0.0
- joblib >= 1.0.0

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available for educational purposes.

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

**Repository**: [https://github.com/dhruvvurhd/ASD](https://github.com/dhruvvurhd/ASD)
