import pandas as pd
import utils as ut

from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split



def load_data(file_path):
    data = pd.DataFrame(ut.load_data(file_path),
                        columns=[
                            "fk_arb_etat",
                            "longitude",
                            "latitude",
                            "haut_tronc",
                            "clc_nbr_diag",
                            "fk_nomtech",
                            "villeca",
                            "feuillage"
                        ]
                        )
    return ut.create_classes_storm(data)


def get_best_model(X, y):
    # Grid Search
    clf = DecisionTreeClassifier()
    param_grid = {
        'criterion': ["gini", "entropy", "log_loss"],
        'random_state': [0, 10, 30, 42],
    }
    # renvoie le meilleur model et le grid search
    return ut.get_best_model(X, y, clf, param_grid)


if __name__ == '__main__':
    d = load_data("../Data_Arbre.csv")
    d = d[d.haut_tot != 0]
    # Séparer les caractéristiques (features) et la cible (target)
    X = d.drop(columns=["fk_arb_etat", "storm_class"])
    y = d["storm_class"]
    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    clf, grid_s, param_grid0 = get_best_model(X_train, y_train)
    ut.print_graph(grid_s, param_grid0, ['criterion', 'random_state'])

    # Examen des meilleurs paramètres et du meilleur modèle
    print("Meilleurs paramètres trouvés : ", grid_s.best_params_)
    print("Meilleur score obtenu : ", grid_s.best_score_)

    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Taux de classification : ", accuracy_score(y_test, y_pred))
    print("Précision (Precision), Rappel (Recall) : \n", classification_report(y_test, y_pred))
    print("Matrice de confusion : \n", confusion_matrix(y_test, y_pred))