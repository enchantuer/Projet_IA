import pandas as pd
import utils as ut
import numpy as np

from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split


import matplotlib.pyplot as plt

def load_data(file_path):
    return pd.DataFrame(ut.load_data(file_path),
                        columns=[
                            "fk_arb_etat",
                            "longitude",
                            "latitude",
                            "clc_secteur",
                            "fk_stadedev",
                            "remarquable"

                        ]
                        )


def create_classes(data):
    # Définir les bornes et les étiquettes des classes
    bins = [0, 1, 2, 3, 4, 5, 6]
    labels = [0, 0, 1, 1, 0, 0]

    # Créer une nouvelle colonne 'AgeClass' pour les classes d'âge
    data["storm_class"] = pd.cut(data["fk_arb_etat"], bins=bins, labels=labels, right=False, ordered=False)

    return data




if __name__ == '__main__':
    d = load_data("../Data_Arbre.csv")
    create_classes(d)
    X = d.drop(columns=["fk_arb_etat", "storm_class"])
    y = d["storm_class"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=10)

    param_grid0 = {

        'max_depth': [6, 8, 20, None],
        'n_estimators': [10, 30, 100]
    }

    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid0, scoring='accuracy')

    grid_search.fit(X_train, y_train)

    ut.print_graph(grid_search, param_grid0,['max_depth','n_estimators'])


    # Examen des meilleurs paramètres et du meilleur modèle
    print("Meilleurs paramètres trouvés : ", grid_search.best_params_)
    print("Meilleur score obtenu : ", grid_search.best_score_)

    clf = clf.fit(X_train, y_train)
    X_pred = clf.predict(X_test)
    print("Taux de classification : ", accuracy_score(y_test, X_pred))
    print("Précision (Precision), Rappel (Recall) : \n", classification_report(y_test, X_pred))
    print("Matrice de confusion : \n", confusion_matrix(y_test, X_pred))