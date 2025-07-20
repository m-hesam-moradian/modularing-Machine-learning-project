from src.data_loader import load_data
from src.preprocess import split_and_scale
from src.model_zoo import get_models
from src.evaluate import evaluate_model
from src.visualize import plot_metrics

# Load and prepare data
X, y = load_data("Data/Fraud Detection Dataset.xlsx", "Fraudulent")
x_train, x_test, y_train, y_test = split_and_scale(X, y)

# Train models and collect results

models = get_models()
results = {}

for name, model in models.items():
    print(f"\n{name} Evaluation:")
    results[name] = evaluate_model(model, x_train, x_test, y_train, y_test)

# Visualization
colors = ['black', 'red', 'yellow', 'orange', 'green', 'blue', 'pink']
for metric in ["train_accuracy", "test_accuracy", "precision", "recall"]:
    plot_metrics(results, metric, colors)
