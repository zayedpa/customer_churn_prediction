"""
Model training module for customer churn prediction
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier, 
    AdaBoostClassifier
)
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import joblib
import warnings
warnings.filterwarnings("ignore")

from preprocessing import (
    load_data, 
    feature_engineering, 
    get_preprocessor, 
    prepare_features
)


def get_models():
    """Define models to train"""
    models = {
        'LogisticRegression': LogisticRegression(max_iter=500),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42),
        'SVC': SVC(probability=True, random_state=42)
    }
    return models


def train_and_compare_models(X_train, y_train, preprocessor):
    """
    Train multiple models and compare using cross-validation
    
    Args:
        X_train: Training features
        y_train: Training target
        preprocessor: Preprocessing pipeline
    Returns:
        results: Dictionary of CV scores per model
    """
    models = get_models()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    
    print("Training models with 5-fold cross-validation...\n")
    for name, model in models.items():
        print(f"Training {name}...")
        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        scores = cross_val_score(
            pipe, X_train, y_train, 
            cv=cv, scoring='roc_auc', n_jobs=1
        )
        results[name] = scores
        print(f"{name:20s} AUC: Mean={scores.mean():.4f} Std={scores.std():.4f}")
    
    return results


def train_best_model(X_train, y_train, preprocessor, results):
    """
    Train the best performing model on full training set
    
    Args:
        X_train: Training features
        y_train: Training target
        preprocessor: Preprocessing pipeline
        results: CV results from train_and_compare_models
    Returns:
        best_pipeline: Trained pipeline
        best_name: Name of best model
    """
    models = get_models()
    best_name = max(results.keys(), key=lambda k: results[k].mean())
    print(f"\n{'='*60}")
    print(f"Best model: {best_name}")
    print(f"CV ROC AUC: {results[best_name].mean():.4f} ± {results[best_name].std():.4f}")
    print(f"{'='*60}\n")
    
    best_model = models[best_name]
    best_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', best_model)
    ])
    
    print("Training best model on full training set...")
    best_pipeline.fit(X_train, y_train)
    print("✓ Training complete!")
    
    return best_pipeline, best_name


def main():
    """Main training pipeline"""
    print("="*60)
    print("Customer Churn Model Training Pipeline")
    print("="*60 + "\n")
    
    # Load and prepare data
    print("Loading data...")
    df = load_data('../data/customer_data.csv')
    
    print("Engineering features...")
    df_fe = feature_engineering(df)
    
    print("Preparing features and target...")
    X, y = prepare_features(df_fe)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Train set: {X_train.shape}")
    print(f"Test set:  {X_test.shape}")
    print(f"Churn rate (train): {y_train.mean():.2%}")
    print(f"Churn rate (test):  {y_test.mean():.2%}\n")
    
    # Get preprocessor
    preprocessor, _, _ = get_preprocessor()
    
    # Train and compare models
    results = train_and_compare_models(X_train, y_train, preprocessor)
    
    # Train best model
    best_pipeline, best_name = train_best_model(
        X_train, y_train, preprocessor, results
    )
    
    # Save model and test data
    print("\nSaving artifacts...")
    joblib.dump(best_pipeline, '../models/best_churn_pipeline.pkl')
    joblib.dump({'X_test': X_test, 'y_test': y_test}, '../models/test_data.pkl')
    print("✓ Model saved: ../models/best_churn_pipeline.pkl")
    print("✓ Test data saved: ../models/test_data.pkl")
    
    print("\n" + "="*60)
    print("Training Complete! Run evaluate.py to see results.")
    print("="*60)
    
    return best_pipeline, X_test, y_test


if __name__ == "__main__":
    main()