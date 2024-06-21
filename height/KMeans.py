import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

import utils as ut
API_KEY = "c1e125d6-b129-4dc0-9622-42cfdcf3ed6e"


def load_data(file_path):
    temp = ut.load_data(file_path, encoder="preprocessing/encode")
    return pd.DataFrame(temp,
                        columns=[
                            "haut_tot",
                            "fk_nomtech",
                            "fk_stadedev",
                            "longitude",
                            "latitude"
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
    ax1.plot([2, 3, 4, 5], ch_list, label="ch", color=color)
    #ax1.tick_params(axis='y', labelcolor=color)

    # Axe 2 : db
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel("sc", color=color)
    ax2.plot([2, 3, 4, 5], sc_list, label="sc", color=color)

    # Axe 3 : sc + db
    ax3 = ax1.twinx()
    color = 'tab:green'
    ax3.set_ylabel("db", color=color)
    ax3.plot([2, 3, 4, 5], db_list, label="db", color=color)

    ax2.set_ylim([0, max(ax2.get_ylim()[1], ax3.get_ylim()[1])])
    ax3.set_ylim(ax2.set_ylim())

    plt.show()


def generate_map(model, XY, X):
    import folium
    from folium.plugins import MarkerCluster

    # URL de la couche de tuiles Stadiamap
    stadiamap_tiles = f"https://tiles.stadiamaps.com/tiles/alidade_smooth_dark/{{z}}/{{x}}/{{y}}.png?api_key={API_KEY}"

    # Coordonnées de Saint-Quentin (centre)
    latitude = 49.8489
    longitude = 3.2876

    # Créer la carte centrée sur Saint-Quentin avec les tuiles Stadiamap
    m = folium.Map(location=[latitude, longitude], zoom_start=13, tiles=None)

    folium.TileLayer(
        tiles=stadiamap_tiles,
        attr='Stadia Maps © OpenMapTiles © OpenStreetMap contributors'
    ).add_to(m)

    marker_cluster_1 = MarkerCluster().add_to(m)
    marker_cluster_2 = MarkerCluster().add_to(m)
    marker_clusters = [marker_cluster_1, marker_cluster_2]

    for i in range(len(X)):
        tree = X.iloc[[i]]
        coord = XY.iloc[[i]]
        cluster = model.predict(tree)[0]
        folium.Marker(
            location=[coord["latitude"], coord["longitude"]],
            icon=folium.Icon(color='blue' if cluster == 0 else 'red')
        ).add_to(marker_clusters[cluster])

    # Sauvegarder la carte dans un fichier HTML
    m.save("output/carte_saint_quentin.html")

    pass


import numpy as np

if __name__ == '__main__':
    d = load_data("Data_Arbre.csv")
    d = d[d.haut_tot != 0]
    # Store coordinate for the map
    xy = pd.DataFrame(d, columns=["longitude", "latitude"])
    # Remove coordinate for the model
    X = d.drop(columns=["longitude", "latitude"])
    # Normalize the data
    X = ut.normalize_datas(X, load_file="preprocessing/norm")
    # Generate the model
    m = generate_model(X, 2)
    print(m.cluster_centers_)
    np.savetxt('models/centroids1.csv', m.cluster_centers_, delimiter=',')

    o = np.loadtxt('models/centroids1.csv', delimiter=',')

    print("cluster :", ut.get_cluster(o, X[:1]))
    print('Vrai cluster :', m.predict(X[:1]))

    # Render the map
    generate_map(m, xy, X)
    #graphics_test(d, [{'k': 2}, {'k': 3}, {'k': 4}, {'k': 5}])
    CH, SC, DB = test_model(m, X)
    print("Calinski-Harabasz : ", CH)
    print("Silhouette : ", SC)
    print("Davies-Bouldin : ", DB)
    pass
