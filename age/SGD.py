import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix

import utils as ut

def load_data(file_path):
    data = pd.DataFrame(ut.load_data(file_path),
                        columns=[
                            "age_estim",
                            "fk_prec_estim",
                            "haut_tot",
                            "fk_stadedev",
                            "tronc_diam"
                        ]
                        )
    return ut.create_classes_age(data)


def get_best_model(X, y):
    # Grid Search
    clf = SGDClassifier()
    param_grid = {
        'loss': ['hinge', 'log_loss', 'modified_huber', 'squared_hinge', 'perceptron'],
        'penalty': ['l2', 'l1', 'elasticnet'],
        'alpha': [0.0001, 0.001, 0.01, 0.1],
    }
    # renvoie le meilleur model et le grid search
    return ut.get_best_model(X, y, clf, param_grid)


if __name__ == "__main__":
    d = load_data("../Data_Arbre.csv")
    d = d[d.haut_tot != 0]
    # Séparer les caractéristiques (features) et la cible (target)
    X = d.drop(columns=["age_estim", "age_class"])
    X = ut.normalize_datas(X, load_file="../preprocessing/norm")
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

    conf_matrix = confusion_matrix(y_test, y_pred)

    # Tracer la matrice de confusion
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["0-19", "20-39", "40-59", "60+"],
                yticklabels=["0-19", "20-39", "40-59", "60+"])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix SGD')
    plt.show()