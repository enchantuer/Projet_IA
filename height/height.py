import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

import utils as ut


def load_data(file_path):
    # TODO : Choix des données à conservé
    temp = ut.load_data(file_path, encoder="../models/norm.pkl")
    return pd.DataFrame(temp,
                        columns=[
                            "haut_tot",
                            "haut_tronc",
                            "fk_nomtech",
                            "fk_stadedev",
                        ]
                        )


def generate_model(data, k):
    model = KMeans(n_clusters=k)
    kmeans = model.fit(data)
    return kmeans


def test_model(model, data):
    ch = calinski_harabasz_score(data, model.predict(data))
    sc = silhouette_score(data, model.predict(data))
    db = davies_bouldin_score(data, model.predict(data))
    return ch, sc, db


def graphics_test(data, parameters):
    ch_list, sc_list, db_list = [], [], []
    train, test = train_test_split(data, test_size=0.5, train_size=0.5)
    for parameter in parameters:
        model = generate_model(train, **parameter)
        ch, sc, db = test_model(model, test)
        ch_list.append(ch)
        sc_list.append(sc)
        db_list.append(db)
    # Plot un graphique
    fig, ax1 = plt.subplots()
    # Axe 1 : ch
    color = 'tab:red'
    ax1.set_xlabel("parameter")
    ax1.set_ylabel("ch", color=color)
    ax1.plot([2,3,4,5], ch_list, label="ch", color=color)
    #ax1.tick_params(axis='y', labelcolor=color)

    # Axe 2 : db
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel("sc", color=color)
    ax2.plot([2,3,4,5], sc_list, label="sc", color=color)

    # Axe 3 : sc + db
    ax3 = ax1.twinx()
    color = 'tab:green'
    ax3.set_ylabel("db", color=color)
    ax3.plot([2,3,4,5], db_list, label="db", color=color)

    ax2.set_ylim([0,max(ax2.get_ylim()[1], ax3.get_ylim()[1])])
    ax3.set_ylim(ax2.set_ylim())

    plt.show()


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
    d = load_data("../Data_Arbre.csv")
    m = generate_model(d, 2)
    ut.save_model(m, "../models/height.pkl")
    #generate_map(m, d)
    #graphics_test(d, [{'k': 2}, {'k': 3}, {'k': 4}, {'k': 5}])
    CH, SC, DB = test_model(m, d)
    print("Calinski-Harabasz : ", CH)
    print("Silhouette : ", SC)
    print("Davies-Bouldin : ", DB)
    pass
