import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

import utils as ut
import numpy as np
from sklearn import model_selection

from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split




def load_data(file_path):
    # TODO : Choix des données à conserver
    data = ut.load_data(file_path)
    return data


def create_classes(data):
    # Définir les bornes et les étiquettes des classes
    bins = [0, 20, 40, 60, np.inf]
    labels = [0, 1, 2, 3]

    # Créer une nouvelle colonne 'AgeClass' pour les classes d'âge
    data["age_class"] = pd.cut(data["age_estim"], bins=bins, labels=labels, right=False)

    return data


def generate_model_age(data):
    # TODO : Création du model
    X = pd.DataFrame(data,
                     columns=[
                         "haut_tot",
                         "haut_tronc",
                         "tronc_diam",
                         "fk_stadedev",
                         "fk_prec_estim",
                         "fk_revetement"
                     ]
                     )
    y= pd.DataFrame(data,
                    columns=[
                        "age_estim"]
                    )

    return X, y




if __name__ == '__main__':
    d = load_data("../Data_Arbre.csv")
    """
    X, y = generate_model_age(d)
    X_train, X_test, y_train, y_test = X[:6000], X[6000:], y[:6000], y[6000:]
    """
    create_classes(d)
    X = d.drop(columns=["age_estim", "age_class"])
    y = d["age_class"]

    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=10)

    param_grid = {
            'n_estimators': [3, 10, 30],
            'max_features': [2, 4, 6, 8]
        }

    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, scoring='accuracy')

    grid_search.fit(X_train, y_train)

    # Examen des meilleurs paramètres et du meilleur modèle
    print("Meilleurs paramètres trouvés : ", grid_search.best_params_)
    print("Meilleur score obtenu : ", grid_search.best_score_)

    clf = clf.fit(X_train, y_train)
    X_pred = clf.predict(X_test)
    print("Taux de classification : ",accuracy_score(y_test, X_pred))
    print("Précision (Precision), Rappel (Recall) : \n",classification_report(y_test, X_pred))
    print("Matrice de confusion : \n",confusion_matrix(y_test, X_pred))




    pass
