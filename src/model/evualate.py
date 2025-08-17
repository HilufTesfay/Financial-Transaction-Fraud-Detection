from sklearn.metrics import average_precision_score, f1_score, confusion_matrix, precision_recall_curve
import matplotlib.pyplot as plt

def evaluate_model(models, X_test, y_test, names):
    """
    Evaluate trained models.
    AUC-PR
    F1 score at 0.5 threshold
    Confusion matrix
    PR curve plot
    models: list of fitted models
    names: list of model names
    """
    results = {}
    for model, name in zip(models, names):
        proba = model.predict_proba(X_test)[:,1]
        ap = average_precision_score(y_test, proba)   
        y_pred = (proba >= 0.5).astype(int)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        results[name] = {"auc_pr": ap, "f1": f1, "cm": cm}

        print(f"\n{name}")
        print(f"AUC-PR: {ap:.4f} | F1@0.5: {f1:.4f}")
        print("Confusion Matrix:\n", cm)

        p, r, _ = precision_recall_curve(y_test, proba)
        plt.figure()
        plt.step(r, p, where="post")
        plt.xlabel("Recall"); plt.ylabel("Precision")
        plt.title(f"PR Curve - {name}")
        plt.show()

    return results
