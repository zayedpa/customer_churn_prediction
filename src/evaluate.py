"""
Model evaluation module for customer churn prediction
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix, 
    classification_report, roc_curve
)
import joblib


def evaluate_model(pipeline, X_test, y_test, model_name="Model"):
    """
    Evaluate trained model on test set
    
    Args:
        pipeline: Trained sklearn pipeline
        X_test: Test features
        y_test: Test target
        model_name: Name for display
    Returns:
        metrics: Dictionary of evaluation metrics
        y_pred: Predictions
        y_proba: Prediction probabilities
    """
    # Predictions
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba)
    }
    
    # Print results
    print("\n" + "="*60)
    print(f"{model_name} - Test Set Evaluation")
    print("="*60)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1']:.4f}")
    print(f"ROC AUC:   {metrics['roc_auc']:.4f}")
    print("="*60)
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))
    
    return metrics, y_pred, y_proba


def plot_confusion_matrix(y_test, y_pred, model_name="Model"):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['No Churn', 'Churn'],
                yticklabels=['No Churn', 'Churn'])
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../models/confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✓ Confusion matrix saved: ../models/confusion_matrix.png")


def plot_roc_curve(y_test, y_proba):
    """Plot and save ROC curve"""
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('../models/roc_curve.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✓ ROC curve saved: ../models/roc_curve.png")


def plot_feature_importance(pipeline, top_n=20):
    """
    Plot feature importances if model supports it
    
    Args:
        pipeline: Trained pipeline
        top_n: Number of top features to show
    """
    classifier = pipeline.named_steps['classifier']
    
    if not hasattr(classifier, 'feature_importances_'):
        print("\n⚠ Model doesn't support feature importances")
        return None
    
    # Get feature names
    preprocessor = pipeline.named_steps['preprocessor']
    num_feats = preprocessor.transformers_[0][2]
    cat_feats = list(
        preprocessor.transformers_[1][1]
        .named_steps['onehot']
        .get_feature_names_out(preprocessor.transformers_[1][2])
    )
    feature_names = num_feats + cat_feats
    
    # Get importances
    importances = classifier.feature_importances_
    fi = pd.Series(importances, index=feature_names).sort_values(ascending=False)[:top_n]
    
    # Plot
    plt.figure(figsize=(10, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(fi)))
    fi.plot(kind='barh', color=colors)
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title(f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('../models/feature_importance.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✓ Feature importance plot saved: ../models/feature_importance.png")
    
    print(f"\nTop {min(10, len(fi))} Most Important Features:")
    for i, (feat, imp) in enumerate(fi.head(10).items(), 1):
        print(f"{i:2d}. {feat:40s} {imp:.4f}")
    
    return fi


def main():
    """Main evaluation pipeline"""
    print("="*60)
    print("Model Evaluation Pipeline")
    print("="*60)
    
    # Load model and test data
    print("\nLoading model and test data...")
    try:
        pipeline = joblib.load('../models/best_churn_pipeline.pkl')
        test_data = joblib.load('../models/test_data.pkl')
        X_test = test_data['X_test']
        y_test = test_data['y_test']
        print("✓ Data loaded successfully")
    except FileNotFoundError:
        print("❌ Error: Model files not found. Please run train_models.py first.")
        return
    
    # Evaluate
    metrics, y_pred, y_proba = evaluate_model(pipeline, X_test, y_test)
    
    # Visualizations
    print("\nGenerating visualizations...")
    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_proba)
    plot_feature_importance(pipeline)
    
    print("\n" + "="*60)
    print("Evaluation Complete!")
    print("="*60)
    
    return metrics


if __name__ == "__main__":
    main()