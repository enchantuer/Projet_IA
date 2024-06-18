import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

import utils as ut
import numpy as np
from sklearn import model_selection

from sklearn.metrics import accuracy_score

def load_data(file_path):
    # TODO : Choix des données à conservé
    data = ut.load_data(file_path)
    return data

def generate_model_age(data):
    # TODO : Création du model
    X = pd.DataFrame(data,
                     columns=[
                         "haut_tot",
                         "haut_tronc",
                         "tronc_diam",
                         "fk_stadedev",
                         "fk_prec_estim"]
                     )
    Y= pd.DataFrame(data,
                    columns=[
                        "age_estim"]
                    )

    return X, Y


if __name__ == '__main__':
    d = load_data("Data_Arbre.csv")
    X, Y = generate_model_age(d)
    X_train, X_test, Y_train, Y_test = X[:6000], X[6000:], Y[:6000], Y[6000:]
    SGD = SGDClassifier()

    SGD.fit(X_train, Y_train)

    print(SGD.predict([X.values[0]]))

    decision_scores = SGD.decision_function([X.values[0]])
    print(decision_scores)

    #accuracy_scores = model_selection.cross_val_score(SGD, X_train, Y_train, cv=3, scoring='accuracy')
    accuracy_scores = accuracy_score(Y_test, SGD.predict(X_test))
    print(accuracy_scores)



    #generate_map(m, d)
    pass
