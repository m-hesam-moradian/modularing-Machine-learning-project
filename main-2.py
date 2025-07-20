from src.data_loader import load_data
from src.preprocess import split_and_scale
from src.model_zoo import model_module
from src.evaluate import evaluate_model
from src.visualize import plot_metrics

# Step 1: Load and prepare data
X, y = load_data("Data/Fraud Detection Dataset.xlsx", "Fraudulent")
x_train, x_test, y_train, y_test = split_and_scale(X, y)

# Step 2: Define all models with config
model_configs = [
    {"name": "GNB", "library": "naive_bayes", "function": "GaussianNB", "attributes": {}},
    {"name": "KNN", "library": "neighbors", "function": "KNeighborsClassifier", "attributes": {}},
    {"name": "DT", "library": "tree", "function": "DecisionTreeClassifier", "attributes": {}},
    {"name": "RF", "library": "ensemble", "function": "RandomForestClassifier", "attributes": {}},
    {"name": "SVM", "library": "svm", "function": "SVC", "attributes": {}},
    {"name": "LR", "library": "linear_model", "function": "LogisticRegression", "attributes": {}},
    {"name": "ANN", "library": "neural_network", "function": "MLPClassifier", "attributes": {}}
]

# Step 3: Build, train, and evaluate all models
results = {}

for config in model_configs:
    model_wrapper = model_module(**config)
    print(f"\n{model_wrapper.name} Evaluation:")
    results[model_wrapper.name] = evaluate_model(
        model_wrapper.model, x_train, x_test, y_train, y_test
    )

# Step 4: Plot metrics
colors = ['black', 'red', 'yellow', 'orange', 'green', 'blue', 'pink']

for metric in ["train_accuracy", "test_accuracy", "precision", "recall"]:
    plot_metrics(results, metric, colors)

