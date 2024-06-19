import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

import utils as ut

def load_data(file_path):
    # TODO : Choix des données à conserver
    return pd.DataFrame(ut.load_data(file_path),
                        columns=[
                            "age_estim",
                            "fk_prec_estim",
                            "haut_tot",
                            "fk_stadedev",
                            "tronc_diam"
                        ]
                        )

def create_classes(data):
    # Définir les bornes et les étiquettes des classes
    bins = [0, 20, 40, 60, np.inf]
    labels = [0, 1, 2, 3]

    # Créer une nouvelle colonne 'AgeClass' pour les classes d'âge
    data["age_class"] = pd.cut(data["age_estim"], bins=bins, labels=labels, right=False)

    return data



d = load_data("../Data_Arbre.csv")
create_classes(d)
# Séparer les caractéristiques (features) et la cible (target)
X = d.drop(columns=["age_estim", "age_class"])
y = d["age_class"]

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

modelSVM = SVC(kernel="linear")

param_grid = {
    'gamma': [1, 0.1, 0.01],
    'kernel': ['linear', 'rbf']
        }

grid_search = GridSearchCV(estimator=modelSVM, param_grid=param_grid, scoring='accuracy')

grid_search.fit(X_train, y_train)

# Examen des meilleurs paramètres et du meilleur modèle
print("Meilleurs paramètres trouvés : ", grid_search.best_params_)
print("Meilleur score obtenu : ", grid_search.best_score_)


modelSVM.fit(X_train, y_train)

y_predSVM = modelSVM.predict(X_test)

print(classification_report(y_test, y_predSVM))
print(confusion_matrix(y_test, y_predSVM))