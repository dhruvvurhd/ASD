# ASD Prediction Model

Machine learning model for predicting Autism Spectrum Disorder (ASD) based on screening questionnaire responses.

## 📄 Documentation

**See [REPORT.md](REPORT.md) for complete analysis and results.**

The REPORT.md file contains:
- Dataset overview and analysis
- Baseline model performance
- Machine learning evaluation results
- Limitations and recommendations

## 🚀 Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Train Model
```bash
python asd_prediction_model.py
```

### Make Predictions
```python
from predict_asd import predict_asd

data = {
    'A1': 1, 'A2': 1, 'A3': 1, 'A4': 1, 'A5': 1,
    'A6': 1, 'A7': 1, 'A8': 1, 'A9': 1, 'A10': 1,
    'Age': 15, 'Sex': 'm', 'Jauundice': 'no', 'Family_ASD': 'yes'
}

results = predict_asd(data)
print(results[0]['ASD_Prediction'])
```

## 📁 Project Structure

- `REPORT.md` - Complete analysis report
- `asd_prediction_model.py` - Training script
- `predict_asd.py` - Prediction interface
- `Autism_Screening_Data_Combined.csv` - Dataset
- `asd_model.pkl` - Trained model
- `requirements.txt` - Dependencies

## 📊 Results Summary

- **Model**: Gradient Boosting
- **Accuracy**: 99.92%
- **Baseline (Simple Rule)**: 93.38%
- **ML Improvement**: +6.54 percentage points

See REPORT.md for detailed analysis.

