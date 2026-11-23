# Customer Churn Prediction

A machine learning project that predicts whether a bank customer is likely to leave (churn), using the `Customer-Churn-Records.csv` dataset of 10,000 customers. Identifying at-risk customers early helps businesses target retention offers and reduce revenue loss.

## Dataset

- **Rows:** 10,000 | **Columns:** 18
- **Target:** `Exited` (1 = customer left, 0 = customer stayed)
- **Features:** CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary, Complain, Satisfaction Score, Card Type, Point Earned
- **Dropped identifiers:** RowNumber, CustomerId, Surname

> **Note on data leakage:** The `Complain` column is almost identical to `Exited` (customers who complained nearly always left). Using it gives ~99.9% accuracy that is not realistic, so it is excluded by default. Set `USE_COMPLAIN = True` in the script to see the effect.

## Project Workflow

1. **Data loading and inspection:** missing values, duplicates, class balance
2. **Exploratory data analysis:** churn by geography, gender, age, products and activity; correlation heatmap
3. **Feature engineering:** balance-to-salary ratio, products per tenure year, age x tenure, zero-balance flag
4. **Preprocessing:** standard scaling for numeric features, one-hot encoding for categorical features (inside a pipeline)
5. **Model training:** Logistic Regression, Random Forest, Gradient Boosting, XGBoost (with class weighting for imbalance)
6. **Evaluation:** 5-fold cross-validation, then accuracy, precision, recall, F1 and ROC-AUC on a held-out test set
7. **Threshold tuning:** finds the probability cutoff with the best F1
8. **Feature importance:** shows the main churn drivers
9. **Deployment-ready:** saves the best model and provides a `predict_churn()` function

## Project Structure

```
.
├── churn_prediction.py          # Main script
├── Customer-Churn-Records.csv   # Dataset
├── README.md
├── churn_model.joblib           # Saved model (generated)
├── eda_plots.png                # EDA charts (generated)
├── correlation_heatmap.png      # Heatmap (generated)
├── model_evaluation.png         # Confusion matrix + ROC curves (generated)
└── feature_importance.png       # Top features (generated)
```

## Installation

```bash
pip install pandas numpy scikit-learn matplotlib seaborn xgboost joblib
```

## Usage

Place the dataset in the same folder as the script, then run:

```bash
python churn_prediction.py
```

### Predicting for a new customer

```python
from churn_prediction import predict_churn

customer = {
    "CreditScore": 600, "Geography": "Germany", "Gender": "Female", "Age": 45,
    "Tenure": 3, "Balance": 120000.0, "NumOfProducts": 1, "HasCrCard": 1,
    "IsActiveMember": 0, "EstimatedSalary": 70000.0, "SatisfactionScore": 2,
    "CardType": "GOLD", "PointEarned": 400,
}

print(predict_churn(customer))
# {'churn_probability': 0.xx, 'will_churn': True/False}
```

> Importing the module re-runs the full training script. For production use, move `predict_churn()` into its own file.

## Results

With `Complain` excluded, expect a ROC-AUC of roughly **0.85-0.87**. Typical top churn drivers:

- Age (older customers churn more)
- Number of products (1 or 3+ products show higher churn)
- Inactive membership
- Geography (Germany shows higher churn)

Exact numbers depend on the random seed and package versions.

## Possible Improvements

- Hyperparameter tuning with `RandomizedSearchCV` or Optuna
- SHAP explanations for individual predictions
- Handle imbalance with SMOTE
- Build a Streamlit or Flask app around the model

## Tech Stack

Python, pandas, NumPy, scikit-learn, XGBoost, Matplotlib, Seaborn, joblib

## License

This project is for educational purposes.
