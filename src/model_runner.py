import importlib
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train_and_evaluate_model(library, function, x_train, x_test, y_train, y_test, attributes=None):
    if attributes is None:
        attributes = {}

    # Load model class dynamically
    module_path = f"sklearn.{library}"
    sklearn_module = importlib.import_module(module_path)
    model_class = getattr(sklearn_module, function)
    model = model_class(**attributes)

    # Train the model
    model.fit(x_train, y_train)

    # Predictions
    y_pred_train = model.predict(x_train)
    y_pred_test = model.predict(x_test)

    # Metrics for train
    acc_train = accuracy_score(y_train, y_pred_train)
    precision_train = precision_score(y_train, y_pred_train)
    recall_train = recall_score(y_train, y_pred_train)
    f1_train = f1_score(y_train, y_pred_train)

    # Metrics for test
    acc_test = accuracy_score(y_test, y_pred_test)
    precision_test = precision_score(y_test, y_pred_test)
    recall_test = recall_score(y_test, y_pred_test)
    f1_test = f1_score(y_test, y_pred_test)

    # Combined metrics
    y_all = np.concatenate([y_train, y_test])
    y_pred_all = np.concatenate([y_pred_train, y_pred_test])

    acc_all = accuracy_score(y_all, y_pred_all)
    precision_all = precision_score(y_all, y_pred_all)
    recall_all = recall_score(y_all, y_pred_all)
    f1_all = f1_score(y_all, y_pred_all)

    # Print metrics
    print(f"[Train] Accuracy: {acc_train:.4f} | Precision: {precision_train:.4f} | Recall: {recall_train:.4f} | F1: {f1_train:.4f}")
    print(f"[Test]  Accuracy: {acc_test:.4f} | Precision: {precision_test:.4f} | Recall: {recall_test:.4f} | F1: {f1_test:.4f}")
    print(f"[All]   Accuracy: {acc_all:.4f} | Precision: {precision_all:.4f} | Recall: {recall_all:.4f} | F1: {f1_all:.4f}")

    return {
        "model": model,
        "train": {
            "accuracy": acc_train,
            "precision": precision_train,
            "recall": recall_train,
            "f1": f1_train,
            "predictions": y_pred_train
        },
        "test": {
            "accuracy": acc_test,
            "precision": precision_test,
            "recall": recall_test,
            "f1": f1_test,
            "predictions": y_pred_test
        },
        "all": {
            "accuracy": acc_all,
            "precision": precision_all,
            "recall": recall_all,
            "f1": f1_all,
        }
    }
