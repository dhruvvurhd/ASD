# ASD Screening Dataset Analysis Report

## 1. Introduction

This report analyzes the **Autism Screening Data (AQ-10 based)** provided for the project. The goal is to understand why machine learning models achieve unusually high accuracy on this dataset and to present a rigorous evaluation of its structure, baseline performance, and model behavior.

The AQ-10 questionnaire is a symptom-based screening tool, and the dataset reflects this structure. Because of this, the dataset is **highly separable**, leading to near-perfect results from even very simple models.

---

## 2. Dataset Overview

* **Total rows:** 6,075
* **Features:** A1–A10 (binary responses), Age, Sex, Jaundice, Family ASD
* **Target variable:** Class (YES/NO)

### 2.1 Class Distribution

| Class | Count | Percentage |
| ----- | ----- | ---------- |
| NO    | 4,271 | ~70%       |
| YES   | 1,804 | ~30%       |

The dataset is moderately imbalanced (~70/30), which already pushes accuracy upward if models default to predicting the majority class.

---

## 3. Dataset Quality Analysis

### 3.1 Duplicate Rows

* **Exact duplicates found:** 1,403
* **Percentage of dataset:** ~23.09%

Duplicates increase the risk of **data leakage** during train/test splits. If identical rows appear in both training and test sets, models appear to "memorize" the dataset.

### 3.2 Feature Pattern Analysis (A1–A10)

The AQ-10 form directly encodes ASD indicators. YES cases tend to have high counts of "1" across these 10 questions.

* Rows where **A1–A10 sum to 10:** 302 rows (~4.97%)
* Correlation between `sum(A1..A10)` and ASD label: **0.704** (very strong)

This means the dataset is **almost linearly separable** using only the summed questionnaire score.

---

## 4. Baseline Model (Non-ML Rule)

Before applying machine learning, a simple threshold on the sum of A1–A10 was tested:

**Rule:** Predict ASD if `sum(A1..A10) ≥ 7`.

| Threshold | Accuracy   |
| --------- | ---------- |
| 7         | **93.38%** |

A single if-statement achieves over 93% accuracy, showing that the dataset is extremely easy to classify.

This becomes the baseline that ML models must outperform.

---

## 5. Machine Learning Evaluation

Multiple machine learning models (Random Forest, Gradient Boosting, SVM, Logistic Regression) were evaluated on stratified 80/20 train/test splits. The best performing model was **Gradient Boosting**.

### 5.1 Evaluation Metrics (Hold-Out Test Set)

Results from the best model (Gradient Boosting):

* **Accuracy:** 99.92% (1,214 out of 1,215 correct)
* **ROC-AUC:** 1.0000
* **Cross-Validation Accuracy:** 99.75% (±0.33%)
* **Precision (YES class):** 100%
* **Recall (YES class):** 100%
* **F1 Score:** 100%

Because the features already encode the target directly, the model simply mirrors the AQ-10 scoring logic. The ML model adds only **+6.54 percentage points** over the simple threshold baseline (93.38%).

### 5.2 Confusion Matrix

```
                Predicted
              NO      YES
Actual NO     853       1
       YES      0     361
```

The confusion matrix shows only 1 misclassification out of 1,215 test samples, confirming the dataset's clean separation. However, this high accuracy is largely due to the dataset structure rather than model sophistication.

---

## 6. Ablation Study

To test which features actually contribute predictive power:

### 6.1 Removing A1–A10

When A1–A10 are removed and only demographic/medical fields remain:

* Model performance collapses heavily.
* This confirms that the questionnaire questions alone drive almost all predictive power.

### 6.2 Using Only A1–A10

Models achieve nearly the same high accuracy, reinforcing that the dataset is essentially a direct encoding of ASD criteria.

---

## 7. Limitations of the Dataset

1. **Trivial separability** – Models do not learn complex patterns; they replicate the AQ-10 scoring rule.
2. **High duplicate rate** – Artificially raises performance.
3. **Screening-based, not clinical** – The dataset does not reflect real diagnostic complexity.
4. **Binary yes/no inputs** – Removes natural variability and noise seen in real assessments.
5. **Potential generalization issues** – Models trained on AQ-10-style data may fail on real clinical populations.

---

## 8. Recommendations

1. **Deduplicate** the dataset before training.
2. Use **F1, precision, recall, ROC-AUC** instead of accuracy.
3. Compare ML models against the **baseline threshold** (sum ≥ 7).
4. Consider additional datasets (NDAR, ABIDE) for robust modeling.
5. Clearly state in the report that high accuracy does **not** imply a clinically reliable model.

---

## 9. Conclusion

The ASD questionnaire dataset used in this project is highly structured and almost perfectly separates ASD vs non-ASD cases based on the A1–A10 responses. This leads to extremely high accuracy from even the simplest models. While the results appear strong, they reflect the simplicity and scoring-rule-like nature of the dataset, not the ability of the model to diagnose ASD in real-world conditions.

This report outlines these findings and provides recommendations to ensure transparent and accurate interpretation of the results.
