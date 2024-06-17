import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

import utils as ut


def load_data(file_path):
    # TODO : Choix des données à conservé
    data = ut.load_data(file_path)
    return data


def generate_model(data, k):
    # TODO : Création du model
    model = KMeans(n_clusters=k)
    X = pd.DataFrame(data,
                     columns=[
                         "haut_tot",
                         "haut_tronc",
                         "fk_nomtech",
                         "fk_stadedev",
                         "age_estim",
                         "fk_prec_estim"]
                     )
    kmeans = model.fit(X)
    return kmeans


def test_model(model, data):
    # TODO : Metric du model
    pass


def generate_map(model, data):
    def plot_clusters(x, y=None):
        plt.scatter(x["longitude"], x["latitude"], c=y, s=1)
        plt.xlabel("longitude", fontsize=14)
        plt.ylabel("latitude", fontsize=14, rotation=0)

    plt.figure()
    X = pd.DataFrame(data,
                     columns=[
                         "haut_tot",
                         "haut_tronc",
                         "fk_nomtech",
                         "fk_stadedev",
                         "age_estim",
                         "fk_prec_estim"]
                     )
    plot_clusters(data, model.predict(X))
    #plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], c="red", s=30)
    plt.show()
    pass


if __name__ == '__main__':
    d = load_data("Data_Arbre.csv")
    m = generate_model(d, 2)
    generate_map(m, d)

    pass
