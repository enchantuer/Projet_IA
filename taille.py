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
    X = data.drop("haut_tot", axis=1).drop("haut_tronc", axis=1)
    kmeans = model.fit(X)

    return kmeans

def test_model(model, data):
    # TODO : Metric du model
    pass

def generate_map(model, data):
    def plot_clusters(X, y=None):
        plt.scatter(X["longitude"], X["latitude"], c=y, s=1)
        plt.xlabel("longitude", fontsize=14)
        plt.ylabel("latitude", fontsize=14, rotation=0)

    plt.figure()
    plot_clusters(data, model.predict(data.drop("haut_tronc", axis=1).drop("haut_tot", axis=1)))
    #plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], c="red", s=30)
    plt.show()
    pass

if __name__ == '__main__':
    data = load_data("Data_Arbre.csv")
    m = generate_model(data, 10)
    generate_map(m, data)

    pass
