from sklearn.cluster import KMeans

import utils as ut

def load_data(file_path):
    # TODO : Choix des données à conservé
    data = ut.load_data(file_path)

    return data

def generate_model(data_path, k):
    # TODO : Création du model
    model = KMeans(n_clusters=k)
    data = load_data(data_path)
    X = data.drop("haut_tot", axis=1).drop("haut_tronc", axis=1)
    kmeans = model.fit(X)

    return kmeans

def test_model(model, data_path):
    # TODO : Metric du model
    pass

def generate_map(model, data_path):
    # TODO : générer la carte avec les clusters sur toutes les données
    pass

m = generate_model("Data_Arbre.csv")