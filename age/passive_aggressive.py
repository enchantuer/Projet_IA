from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

import pandas as pd
import utils as ut


def load_data(file_path):
    data = ut.load_data(file_path)
    return ut.create_classes(data)


def get_best_model(X, y):
    # Grid Search
    clf = PassiveAggressiveClassifier(random_state=42)
    param_grid = {
        'max_iter': [500, 1000, 2000],
        'C': [1, 2, 5, 10, 50]
    }
    # renvoie le meilleur model et le grid search
    return ut.get_best_model(X, y, clf, param_grid)


if __name__ == '__main__':
    d = load_data("../Data_Arbre.csv")
    # Séparer les caractéristiques (features) et la cible (target)
    X = d.drop(columns=["age_estim", "age_class"])
    y = d["age_class"]
    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    clf, grid_s = get_best_model(X_train, y_train)

    # Examen des meilleurs paramètres et du meilleur modèle
    print("Meilleurs paramètres trouvés : ", grid_s.best_params_)
    print("Meilleur score obtenu : ", grid_s.best_score_)

    y_pred = clf.predict(X_test)

    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
