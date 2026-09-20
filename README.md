# Credit Card Default Classification in Python

## Overview

This project predicts credit card default using Python and compares four classification approaches:

- L1 Logistic Regression
- Bagging
- Random Forest
- Gradient Boosting

The project was designed as a modular and reproducible machine-learning workflow rather than a single analysis script. Data preprocessing, model training, and model evaluation are separated into reusable Python modules.

## Dataset

The project uses the public **UCI Credit Card Default dataset**, containing 30,000 customer records and information on:

- Credit limits
- Demographic characteristics
- Repayment history
- Bill amounts
- Previous payments
- Default status

The target variable indicates whether a customer defaulted on the next payment.

## Project Structure

```text
credit-card-default-classification-python/
│
├── data/
│   └── UCI_Credit_Card.csv
│
├── notebooks/
│   └── credit_card_default_analysis.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── modeling.py
│   └── evaluation.py
│
├── results/
├── requirements.txt
├── .gitignore
└── README.md
```

## Workflow

```text
Raw Data
   ↓
Data Validation
   ↓
Cleaning and Recoding
   ↓
Stratified Train/Test Split
   ↓
Feature Preprocessing
   ↓
Machine-Learning Pipelines
   ↓
Model Training
   ↓
Model Evaluation
```

The dataset was divided into:

- **70% training data**
- **30% test data**

A stratified split was used to preserve the proportion of default and non-default cases.

## Machine-Learning Models

### L1 Logistic Regression

L1 regularization was combined with 10-fold cross-validation to control model complexity and perform feature selection.

### Bagging

Multiple decision trees were trained on bootstrap samples and their predictions were combined to reduce model variance.

### Random Forest

Random Forest extends bagging by also selecting random subsets of predictors when constructing individual trees.

### Gradient Boosting

Gradient Boosting builds trees sequentially, with later trees attempting to improve errors made by earlier trees.

## Model Evaluation

Because credit card default is an imbalanced classification problem, model performance was evaluated using:

- Accuracy
- Recall
- Specificity
- Precision
- F1 Score

## Results

| Model | Accuracy | Recall | Specificity | Precision | F1 |
|---|---:|---:|---:|---:|---:|
| L1 Logistic Regression | 0.8097 | 0.2275 | 0.9750 | 0.7213 | 0.3459 |
| Bagging | 0.8176 | **0.3787** | 0.9422 | 0.6506 | **0.4787** |
| Random Forest | 0.8172 | 0.3666 | 0.9452 | 0.6553 | 0.4702 |
| Gradient Boosting | **0.8240** | 0.3606 | 0.9556 | 0.6978 | 0.4755 |

Gradient Boosting achieved the highest overall accuracy.

Bagging achieved the highest recall and F1 score, making it particularly useful when identifying actual default cases is a priority.

This comparison shows why model selection should consider the analytical objective rather than relying only on accuracy.

## Python Skills Demonstrated

- pandas data manipulation
- Data validation and preprocessing
- Train/test splitting
- Stratified sampling
- One-hot encoding
- Feature standardization
- scikit-learn Pipelines
- L1-regularized logistic regression
- Ensemble machine learning
- Cross-validation
- Model evaluation
- Confusion matrices
- Modular Python development
- Reproducible analytical workflows

## Technologies

- Python
- pandas
- NumPy
- scikit-learn
- matplotlib
- Jupyter Notebook
- Git
- GitHub

## Reproducibility

Install the required packages using:

```bash
python -m pip install -r requirements.txt
```

The executed Jupyter notebook contains the model outputs, comparison table, visualizations, and confusion matrices.

## Automated Testing and CI

The project includes automated unit tests for:

- Data preprocessing
- Train/test splitting
- Model construction
- Model fitting
- Evaluation metrics

Tests are written with **pytest** and run automatically through **GitHub Actions** on pushes and pull requests to the `main` branch.

Current test status:

- 6 automated tests
- Preprocessing tests
- Modeling tests
- Evaluation tests
- GitHub Actions CI workflow
