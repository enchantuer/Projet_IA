from sklearn.metrics import accuracy_score, classification_report, accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
import utils as ut



def load_data(file_path):
    data = pd.DataFrame(ut.load_data(file_path),
                        columns=[
                            "haut_tot",
                            "haut_tronc",
                            "tronc_diam",
                            "fk_stadedev",
                            "fk_prec_estim",
                            "fk_revetement",
                            "age_estim"
                        ]
                        )
    return ut.create_classes_age(data)


def get_best_model(X, y):
    # Grid Search
    clf = RandomForestClassifier()
    param_grid = {
            'max_features': [2, 4, 6, 8],
            'n_estimators': [3, 10, 30]

    }
    # renvoie le meilleur model et le grid search
    return ut.get_best_model(X, y, clf, param_grid)


if __name__ == '__main__':
    d = load_data("../Data_Arbre.csv")
    d = d[d.haut_tot != 0]
    # Séparer les caractéristiques (features) et la cible (target)
    X = d.drop(columns=["age_estim", "age_class"])
    X = ut.normalize_datas(X, load_file="../preprocessing/norm")
    y = d["age_class"]
    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    clf, grid_s, param_grid0 = get_best_model(X_train, y_train)

    ut.print_graph(grid_s, param_grid0, ['n_estimators', 'max_features'])

    # Examen des meilleurs paramètres et du meilleur modèle
    print("Meilleurs paramètres trouvés : ", grid_s.best_params_)
    print("Meilleur score obtenu : ", grid_s.best_score_)

    y_pred = clf.predict(X_test)
    print("Taux de classification : ",accuracy_score(y_test, y_pred))
    print("Précision (Precision), Rappel (Recall) : \n",classification_report(y_test, y_pred))
    print("Matrice de confusion : \n",confusion_matrix(y_test, y_pred))

    conf_matrix = confusion_matrix(y_test, y_pred)

    # Tracer la matrice de confusion
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["0-19", "20-39", "40-59", "60+"],
                yticklabels=["0-19", "20-39", "40-59", "60+"])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix Random Forest')
    plt.show()
