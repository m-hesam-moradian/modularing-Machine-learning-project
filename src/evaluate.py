from sklearn.metrics import accuracy_score, precision_score, recall_score

def evaluate_model(model, x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train)
    y_pred_train = model.predict(x_train)
    y_pred_test = model.predict(x_test)

    acc_train = accuracy_score(y_train, y_pred_train)
    acc_test = accuracy_score(y_test, y_pred_test)
    precision = precision_score(y_test, y_pred_test)
    recall = recall_score(y_test, y_pred_test)

    print(f"Train Accuracy: {acc_train:.4f} - Test Accuracy: {acc_test:.4f} - Precision: {precision:.4f} - Recall: {recall:.4f}")
    
    return {
        "train_accuracy": acc_train,
        "test_accuracy": acc_test,
        "precision": precision,
        "recall": recall
    }
