import pandas as pd
import utils as ut
import numpy as np

from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV, train_test_split



def load_data(file_path):
    # TODO : Choix des données à conservé
    data = ut.load_data(file_path)
    return data


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
    clf = GaussianNB()

    param_grid = {
        #aucun paramètre à optimiser
    }

    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, scoring='accuracy')

    grid_search.fit(X_train, y_train)

    # Examen des meilleurs paramètres et du meilleur modèle
    print("Meilleurs paramètres trouvés : ", grid_search.best_params_)
    print("Meilleur score obtenu : ", grid_search.best_score_)

    clf.fit(X_train, y_train)
    X_pred = clf.predict(X_test)
    print("Taux de classification : ", accuracy_score(y_test, X_pred))
    print("Précision (Precision), Rappel (Recall) : \n", classification_report(y_test, X_pred))
    print("Matrice de confusion : \n", confusion_matrix(y_test, X_pred))
