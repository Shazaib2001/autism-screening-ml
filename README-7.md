# Early Autism Screening via Behavioural Data — Machine Learning Classification Model

![Python](https://img.shields.io/badge/Python-3.10-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Platform](https://img.shields.io/badge/Platform-Google%20Colab-orange)

---

## Overview

Autism Spectrum Disorder (ASD) is significantly underdiagnosed, particularly 
in younger children and females. Early identification is critical — research 
consistently shows that earlier intervention leads to substantially better 
developmental outcomes. Yet in practice, formal diagnosis can take years due 
to long waiting lists and limited clinical resources.

This project builds a machine learning classification model trained on 
behavioural questionnaire responses and clinical data to predict whether a 
child is likely to have ASD traits. The goal is not to replace clinical 
diagnosis but to provide a fast, accessible screening tool that can help 
prioritise which children need urgent assessment.

> **Important:** This is a screening tool, not a diagnostic tool. A positive 
> prediction should trigger referral for formal clinical assessment — not serve 
> as a diagnosis in itself.

---

## Key Question

Can behavioural questionnaire responses and clinical features predict ASD 
traits in children with sufficient accuracy to support early screening 
decisions?

---

## Key Findings

### Model Performance
Random Forest achieved 89% accuracy on unseen test data using purely 
behavioural and clinical features — outperforming the Q-Chat screening 
tool's reported benchmark of 80-85%.

### Most Predictive Features
| Rank | Feature | Importance | Clinical Meaning |
|------|---------|------------|-----------------|
| 1 | Qchat_10_Score | 0.41 | Total behavioural questionnaire score |
| 2 | A7 | 0.11 | Distress when routines change |
| 3 | A6 | 0.10 | Difficulty following joint attention |
| 4 | Family_mem_with_ASD | 0.07 | Family history of ASD |
| 5 | A9 | 0.05 | Purposeless gazing behaviour |

### SHAP Insights
- High Qchat scores push strongly towards ASD prediction
- Answering yes to behavioural questions (A6, A7, A9) consistently 
  increases ASD prediction probability
- Family history meaningfully increases prediction probability, 
  validating the well-established genetic component of autism
- Comorbidities (depression, anxiety, social/behavioural issues) 
  play a supporting but secondary role

---

## Dataset

- **Source:** Autism Screening dataset
- **Size:** 1,985 patient records (1,923 after cleaning)
- **Features:** 28 columns including behavioural questionnaire responses, 
  clinical scores and demographic information
- **Target variable:** ASD_traits (0 = No ASD traits, 1 = ASD traits)

| Metric | Value |
|--------|-------|
| Total patients | 1,985 |
| After cleaning | 1,923 |
| ASD positive | 1,018 (53%) |
| ASD negative | 905 (47%) |
| Features after engineering | 20 |

---

## Pipeline

```
Raw Data (.csv)
    │
    ▼
Data Cleaning
(missing values, duplicates, irrelevant columns)
    │
    ▼
Encoding
(Label Encoding, One-Hot Encoding)
    │
    ▼
Exploratory Data Analysis
(histograms, correlation heatmap, feature vs target, class distribution)
    │
    ▼
Feature Engineering
(dropped weak, biased and demographic features)
    │
    ▼
Train Test Split (80/20)
    │
    ▼
Preprocessing (RobustScaler)
    │
    ▼
Class Balancing (SMOTE)
    │
    ▼
Hyperparameter Tuning (RandomizedSearchCV)
    │
    ▼
Model Training
(Random Forest | Logistic Regression)
    │
    ▼
Evaluation
(accuracy, precision, recall, F1, confusion matrix)
    │
    ▼
Feature Importance + SHAP Analysis
```

---

## Models

Two models were trained and compared:

| Metric | Random Forest | Logistic Regression |
|--------|--------------|---------------------|
| Accuracy | 89% | 81% |
| Precision (ASD) | 1.00 | 0.95 |
| Recall (ASD) | 0.79 | 0.70 |
| F1 Score | 0.89 | 0.80 |
| Cross-validated F1 | 0.92 | 0.86 |
| Missed ASD cases | 44 | 65 |

**Random Forest was selected as the primary model** based on higher recall, 
precision and cross-validation consistency. For a clinical screening tool, 
recall is the most critical metric — missing an ASD case means a child 
misses the early intervention they need.

---

## Ethical Considerations

Sex and ethnicity features were deliberately removed from the final model 
despite improving performance metrics. Two reasons drove this decision:

**Sex:** Males are diagnosed with ASD at roughly a 4:1 ratio compared to 
females in clinical data. Including sex as a feature would cause the model 
to systematically underscreen females — perpetuating an existing problem 
where autism in females is already significantly underdiagnosed due to 
different symptom presentation and masking behaviours.

**Ethnicity:** Correlation between ethnicity and ASD traits in this dataset 
reflects the demographic composition of the data collection, not genuine 
clinical relationships. Using ethnicity as a predictor would introduce 
unjustifiable bias into a clinical tool.

A screening tool must perform equitably across all children regardless of 
gender or background. Performance was prioritised only after fairness was 
ensured.

---

## Limitations

| Limitation | Impact |
|------------|--------|
| Single dataset | No external validation — results may not generalise across populations |
| Recall of 0.79 | 21% of ASD cases are missed on test data |
| Age distribution skew | Model may perform better for ages 7-15 than younger toddlers |
| Dataset size | 1,923 patients is modest — larger datasets would improve robustness |
| Self-reported features | Q-Chat responses are parent-reported and subject to bias |

---

## Future Directions

1. **External validation** — test on independent autism screening datasets 
   to assess generalisability across different populations and settings

2. **Improve recall** — investigate cost-sensitive learning or threshold 
   adjustment to reduce missed ASD cases below 10%

3. **Younger children** — collect more data for children under 5 where 
   early screening has the greatest clinical impact

4. **Female ASD presentation** — incorporate datasets with higher female 
   representation to improve screening equity

5. **Longitudinal validation** — track whether children flagged by the 
   model go on to receive formal ASD diagnoses

---

## Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn shap scipy
```

---

## Usage

1. Open `autism_screening_ml.ipynb` in Google Colab or Jupyter Notebook
2. Upload your dataset file when prompted
3. Run all cells in order from top to bottom
4. Results, plots and SHAP analysis will render inline

---

## Author

**Elvis**  
Machine Learning | Health Data Science

[LinkedIn](#) | [GitHub](#)

---

## License

This project is licensed under the MIT License — see the LICENSE file 
for details.

---

## Acknowledgements

- Dataset sourced from the UCI Autism Screening repository
- SHAP library for model explainability
- imbalanced-learn for SMOTE implementation
