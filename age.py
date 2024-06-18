import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

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