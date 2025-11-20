# Customer Churn Prediction Project

A complete machine learning project to predict customer churn using various classification algorithms.

## 📁 Project Structure
customer_churn_project/
├── data/                     # Raw dataset
│   └── customer_data.csv
├── notebooks/                # Jupyter notebooks for exploration
│   ├── EDA.ipynb            # Exploratory data analysis
│   └── Modeling.ipynb       # Model experimentation
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