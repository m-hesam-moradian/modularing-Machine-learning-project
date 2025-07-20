import pandas as pd

def load_data(path: str, target_column: str):
    data = pd.read_excel(path, engine='openpyxl')
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    return X, y
