# Customer Churn Prediction Using Machine Learning

A machine learning project that predicts whether a telecom customer is likely to leave a company based on customer demographics, subscribed services, contract details, tenure, and billing information.

## Overview

Customer churn is a major challenge for businesses that rely on recurring customers. Identifying customers who may leave can help companies develop better customer retention strategies.

This project uses supervised machine learning to classify customers into two categories:

- **Churn = Yes:** The customer left the company.
- **Churn = No:** The customer stayed with the company.

The project covers data preprocessing, exploratory data analysis, model training, evaluation, feature importance, and customer churn prediction.

## Objectives

- Analyze customer data to understand churn patterns.
- Clean missing values and preprocess numerical and categorical features.
- Train and compare machine learning classification models.
- Evaluate model performance using multiple metrics.
- Identify features that influence model predictions.
- Save the trained model for future predictions.

## Dataset

The project uses a telecom customer churn dataset containing **7,043 records and 21 columns**.

Important features include:

| Feature | Description |
|---|---|
| `tenure` | Number of months the customer has stayed |
| `Contract` | Customer's contract type |
| `MonthlyCharges` | Monthly bill amount |
| `TotalCharges` | Total accumulated charges |
| `InternetService` | Type of internet service |
| `PaymentMethod` | Customer's payment method |
| `TechSupport` | Whether technical support is included |
| `Churn` | Target variable |

The target column is `Churn`, where `Yes` represents churn and `No` represents a customer who stayed.

> **Dataset note:** Include the dataset in this repository only if its license and source terms permit redistribution. Add the original dataset source and attribution here.

## Technologies Used

- **Python** — Programming language
- **Pandas** — Data manipulation
- **NumPy** — Numerical computations
- **Matplotlib** — Data visualization
- **Seaborn** — Exploratory data analysis
- **Scikit-learn** — Machine learning and model evaluation
- **Jupyter Notebook** — Interactive development
- **Joblib** — Saving and loading the trained model

## Machine Learning Workflow

1. **Data loading:** Import the dataset using Pandas.
2. **Data inspection:** Examine data types, missing values, duplicate rows, and target distribution.
3. **Exploratory Data Analysis:** Visualize churn distribution and relationships involving contract type, tenure, and monthly charges.
4. **Data preprocessing:** Handle missing values, scale numerical features, and one-hot encode categorical features.
5. **Feature and target separation:** Remove the customer identifier and separate the input features from the churn label.
6. **Train-test split:** Use 80% of the data for training and 20% for testing, with stratification.
7. **Model training:** Train Logistic Regression and Random Forest classifiers.
8. **Model evaluation:** Compare performance using classification metrics and visualizations.
9. **Feature analysis:** Examine model coefficients or feature importance.
10. **Prediction and saving:** Demonstrate a prediction and save the selected model pipeline.

## Machine Learning Algorithms

### 1. Logistic Regression

A classification algorithm that estimates the probability of customer churn. It provides a useful baseline and can offer interpretable coefficients.

### 2. Random Forest Classifier

An ensemble learning algorithm that combines predictions from multiple decision trees. It can capture nonlinear patterns and interactions between features.

Both models are evaluated on the same held-out test set.

## Evaluation Metrics

The project evaluates the models using:

- **Accuracy:** Overall proportion of correct predictions.
- **Precision:** Proportion of predicted churners who actually churned.
- **Recall:** Proportion of actual churners correctly identified.
- **F1-score:** Harmonic mean of precision and recall.
- **ROC-AUC:** Ability to distinguish churners from non-churners across classification thresholds.

Recall and precision are particularly important because the dataset contains fewer churners than non-churners.

## Project Structure
```
customer-churn-prediction/
│
├── README.md
├── churn_datafile.csv
├── Customer_Churn_Prediction_Project.ipynb
├── requirements.txt
│
├── images/
│   ├── churn_distribution.png
│   ├── confusion_matrix.png
│   └── roc_curve.png
│
└── models/
    └── customer_churn_model.joblib
```

## Results

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7381 | 0.5043 | 0.7834 | 0.6136 | 0.8413 |
| Random Forest | 0.7622 | 0.5394 | 0.7139 | 0.6145 | 0.8355 |

The best model should be selected based on the measured evaluation results and the project's business objective. Do not fill this table with estimated or unverified scores.

## Model Prediction

The notebook demonstrates how to predict churn for a customer using the trained pipeline.

The pipeline can also be saved for later use:

```python
import joblib

joblib.dump(best_model, "customer_churn_model.joblib")

loaded_model = joblib.load("customer_churn_model.joblib")
```

The saved model accepts customer features in the expected input format and returns a churn prediction. The probability is an estimate, not a guarantee that the customer will leave.

## Future Improvements

- Tune hyperparameters using cross-validation.
- Optimize the prediction threshold based on business costs.
- Compare additional algorithms such as Gradient Boosting.
- Add SHAP-based model explanations.
- Build an interactive web application using Streamlit.
- Monitor model performance on new customer data.

## Limitations

- The model learns from historical data and may not generalize perfectly to future customers.
- The dataset has an imbalanced target distribution.
- Feature importance indicates predictive relevance, not causation.
- Model performance and business usefulness must be verified through testing.

## Author

**Aman Shivawedy**

- GitHub: (https://github.com/amannn05/customer-churn-prediction)
- LinkedIn: (https://www.linkedin.com/in/aman-shivawedy-72a8b5324/?isSelfProfile=true)
