from sklearn.cluster import KMeans
import pandas as pd

def load_data(file_path):
    # TODO : Traitement des données
    return pd.read_csv(file_path)

def generate_model(data_path, k):
    model = KMeans(n_clusters=k)
    data = load_data(data_path)
    X = data.drop("haut_tot", axis=1).drop("haut_tronc", axis=1)
    Y = data["haut_tot"].copy()

    kmeans = model.fit(X)

m = generate_model("Data_Arbre.csv")