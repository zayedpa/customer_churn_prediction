## Customer Churn Prediction Project

A complete end-to-end machine learning project to analyze customer behavior and predict churn for a bank.
The project includes data preprocessing, feature engineering, model training, evaluation, and insights using explainable AI techniques.


## Project Highlights

Built ML models to predict whether a customer will churn
Performed data cleaning & preprocessing
Applied feature engineering (age bucket, balance ratios, etc.)
Trained multiple models (XGBoost / RandomForest / Logistic Regression)
Used Explainable AI: Feature Importance, ROC Curve, SHAP values
Achieved AUC = 0.87
Included Confusion Matrix to analyze model errors

## 📁 Project Structure
customer_churn_project/
├── data/                    
│   └── customer_data.csv
├── notebooks/                
│   ├── EDA.ipynb             
├── src/                      # Core Python scripts
│   ├── preprocessing.py     # Data cleaning and feature engineering
│   ├── train_models.py      # Model training pipeline
│   └── evaluate.py          # Model evaluation
├── models/                   # Trained model artifacts
│   └── best_churn_pipeline.pkl
├── app/                      # Deployment
│   └── streamlit_app.py     # Interactive web app
├── README.md
└── requirements.txt


## Tech Stack
Python
Pandas, NumPy
Scikit-Learn
Matplotlib,Seaborn
Streamlit
joblib
VS code

ML Algorithms: Logistic Regression, Random Forest, Gradient Boosting, AdaBoost, SVM
