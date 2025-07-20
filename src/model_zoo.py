from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import importlib


# Optional: Predefined models dictionary
def get_models():
    return {
        "GNB": GaussianNB(),
        "KNN": KNeighborsClassifier(),
        "DT": DecisionTreeClassifier(),
        "RF": RandomForestClassifier(),
        "SVM": SVC(),
        "LR": LogisticRegression(),
        "ANN": MLPClassifier()
    }


# Wrapper class to hold model and name
class ModelWrapper:
    def __init__(self, name, library, function, attributes=None):
        if attributes is None:
            attributes = {}

        # Dynamically import the module
        module_path = f"sklearn.{library}"
        sklearn_module = importlib.import_module(module_path)

        # Get the model class
        model_class = getattr(sklearn_module, function)

        self.name = name
        self.model = model_class(**attributes)


# Main model module function
def model_module(name, library, function, attributes=None):
    return ModelWrapper(name, library, function, attributes)
