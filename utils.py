import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
import skops.io as sio
from sklearn.model_selection import GridSearchCV
import numpy as np


def normalise_data(data, column, load_file=None, path_to_save=None):
    train_cat = data[[column]]
    if load_file is None:
        temp = OrdinalEncoder()
        temp.fit(train_cat)
    else:
        temp = load_model(load_file)
    if path_to_save is not None:
        save_model(temp, path_to_save)
    data[column] = temp.transform(train_cat)

    return data


def load_data(file_path, encoder=None, path_to_save_encoder=None):
    data = pd.read_csv(file_path)
    liste_modif = ["clc_quartier", "clc_secteur", "fk_arb_etat", "fk_stadedev", "fk_port", "fk_pied",
                   "fk_situation", "fk_revetement", "fk_nomtech", "villeca", "feuillage", "remarquable"]
    for col in liste_modif:
        if encoder is None:
            normalise_data(data, col, None, path_to_save_encoder+"/"+col+".pkl")
        else:
            normalise_data(data, col, encoder+"/"+col+".pkl", path_to_save_encoder)
    return data

def load_model(file_name):
    unknown_types = sio.get_untrusted_types(file=file_name)
    return sio.load(file_name, trusted=unknown_types)


def save_model(model, file_name):
    sio.dump(model, file=file_name)


def get_best_model(X, y, model, params):
    # Grid Search
    grid_search = GridSearchCV(estimator=model, param_grid=params, scoring='accuracy')
    # Fit les models
    grid_search.fit(X, y)
    # renvoie le meilleur model et le grid search
    return grid_search.best_estimator_, grid_search


def create_classes(data):
    # Définir les bornes et les étiquettes des classes
    bins = [0, 20, 40, 60, np.inf]
    labels = [0, 1, 2, 3]

    # Créer une nouvelle colonne 'AgeClass' pour les classes d'âge
    data["age_class"] = pd.cut(data["age_estim"], bins=bins, labels=labels, right=False)

    return data


if __name__ == '__main__':
    print(load_data("Data_Arbre.csv", encoder="norm", path_to_save_encoder=None))
