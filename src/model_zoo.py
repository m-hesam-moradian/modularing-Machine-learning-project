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

def model_module(library, function, attributes=None):
    if attributes is None:
        attributes = {}
    module_path = f"sklearn.{library}"
    sklearn_module = importlib.import_module(module_path)
    model_class = getattr(sklearn_module, function)
    return model_class(**attributes)  # ⬅️ مستقیماً مدل رو برمی‌گردونیم
