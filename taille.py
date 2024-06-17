import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

import utils as ut


def load_data(file_path):
    # TODO : Choix des données à conservé
    return pd.DataFrame(ut.load_data(file_path),
                        columns=[
                            "haut_tot",
                            "haut_tronc",
                            "fk_nomtech",
                            "fk_stadedev",
                            "age_estim",
                            "fk_prec_estim",
                            "longitude",
                            "latitude"
                        ]
                        )


def generate_model(data, k):
    model = KMeans(n_clusters=k)
    kmeans = model.fit(data)
    return kmeans


def test_model(model, data):
    ch = calinski_harabasz_score(data, model.labels_)
    sc = silhouette_score(data, model.labels_)
    db = davies_bouldin_score(data, model.labels_)
    return ch, sc, db


def generate_map(model, data):
    def plot_clusters(x, y=None):
        plt.scatter(x["longitude"], x["latitude"], c=y, s=1)
        plt.xlabel("longitude", fontsize=14)
        plt.ylabel("latitude", fontsize=14, rotation=0)

    plt.figure()
    plot_clusters(data, model.labels_)
    #plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], c="red", s=30)
    plt.show()
    pass


if __name__ == '__main__':
    d = load_data("Data_Arbre.csv")
    m = generate_model(d, 2)
    generate_map(m, d)
    CH, SC, DB = test_model(m, d)
    print("Calinski-Harabasz : ", CH)
    print("Silhouette : ", SC)
    print("Davies-Bouldin : ", DB)
    pass
