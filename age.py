import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

import utils as ut


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

    return kmeans


if __name__ == '__main__':
    d = load_data("Data_Arbre.csv")
    m = generate_model(d, 2)
    #generate_map(m, d)
    graphics_test(d, [{'k': 2}, {'k': 3}, {'k': 4}, {'k': 5}])
    CH, SC, DB = test_model(m, d)
    print("Calinski-Harabasz : ", CH)
    print("Silhouette : ", SC)
    print("Davies-Bouldin : ", DB)
    pass
