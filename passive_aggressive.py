from sklearn.svm import SVC

from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, classification_report

import utils as ut
import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV, train_test_split






def load_data(file_path):
    # TODO : Choix des données à conservé
    data = ut.load_data(file_path)
    return data


def create_classes(data):
    # Définir les bornes et les étiquettes des classes
    bins = [0, 20, 40, 60, np.inf]
    labels = [0, 1, 2, 3]

    # Créer une nouvelle colonne 'AgeClass' pour les classes d'âge
    data["age_class"] = pd.cut(data["age_estim"], bins=bins, labels=labels, right=False)

    return data



if __name__ == '__main__':
    d = load_data("Data_Arbre.csv")
    create_classes(d)
    X = d.drop(columns=["age_estim", "age_class"])
    y = d["age_class"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pac = PassiveAggressiveClassifier(max_iter=1000, random_state=42)
    pac.fit(X_train, y_train)
    y_pred = pac.predict(X_test)

    # Calcul de l'accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy : ",accuracy)

    # Calcul des information importantes
    classificationReport=classification_report(y_test, y_pred)
    print("Classification Report : \n",classificationReport)