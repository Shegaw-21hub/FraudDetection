import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, precision_recall_curve, auc, average_precision_score

def evaluate_model(model, X_test, y_test, model_name, dataset_name):
    """
    Evaluates a trained model and prints performance metrics suitable for imbalanced data.
    Plots the Precision-Recall Curve.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_proba)
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    auc_pr = auc(recall, precision)
    avg_precision = average_precision_score(y_test, y_proba)

    print(f"\n--- {model_name} on {dataset_name} Evaluation Summary ---")
    print(f"F1-Score: {f1:.4f}")
    print(f"AUC-ROC: {auc_roc:.4f}")
    print(f"AUC-PR: {auc_pr:.4f}")
    print(f"Confusion Matrix:\n{cm}")

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'Precision-Recall curve (AUC-PR = {auc_pr:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve for {model_name} on {dataset_name}')
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.show()

    return {'F1': f1, 'AUC-ROC': auc_roc, 'AUC-PR': auc_pr, 'Confusion Matrix': cm}
