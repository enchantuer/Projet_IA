import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
import skops.io as sio
from sklearn.model_selection import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt


def encode_data(data, column, load_file=None, path_to_save=None):
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


def normalize_data(data, column, load_file=None, path_to_save=None):
    train_cat = data[[column]]
    if load_file is None:
        normalizer = MinMaxScaler()
        normalizer.fit(train_cat)
    else:
        normalizer = load_model(load_file)
    if path_to_save is not None:
        save_model(normalizer, path_to_save)
    data[column] = normalizer.transform(train_cat)
    return data


def normalize_datas(data, load_file=None, path_to_save=None):
    for col in data.columns:
        if load_file is None:
            if path_to_save is None:
                normalize_data(data, col, None, None)
            else:
                normalize_data(data, col, None, path_to_save + "/norm_" + col + ".pkl")
        else:
            if path_to_save is None:
                normalize_data(data, col, load_file + "/norm_" + col + ".pkl", None)
            else:
                normalize_data(data, col, load_file + "/norm_" + col + ".pkl", path_to_save + "/norm_" + col + ".pkl")
    return data


def load_data(file_path, encoder=None, path_to_save_encoder=None):
    data = pd.read_csv(file_path)
    liste_modif = ["clc_quartier", "clc_secteur", "fk_arb_etat", "fk_stadedev", "fk_port", "fk_pied",
                   "fk_situation", "fk_revetement", "fk_nomtech", "villeca", "feuillage", "remarquable"]
    for col in liste_modif:
        if encoder is None:
            if path_to_save_encoder is None:
                encode_data(data, col, None, None)
            else:
                encode_data(data, col, None, path_to_save_encoder + "/encode_" + col + ".pkl")
        else:
            if path_to_save_encoder is None:
                encode_data(data, col, encoder + "/encode_" + col + ".pkl", None)
            else:
                encode_data(data, col, encoder + "/encode_" + col + ".pkl", path_to_save_encoder + "/encode_" + col + ".pkl")
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
    return grid_search.best_estimator_, grid_search, params


def create_classes_age(data):
    # Définir les bornes et les étiquettes des classes
    bins = [0, 20, 40, 60, np.inf]
    labels = [0, 1, 2, 3]

    # Créer une nouvelle colonne 'AgeClass' pour les classes d'âge
    data["age_class"] = pd.cut(data["age_estim"], bins=bins, labels=labels, right=False)

    return data

def create_classes_storm(data):
    # Définir les bornes et les étiquettes des classes
    bins = [0, 1, 2, 3, 4, 5, 6]
    labels = [0, 0, 1, 1, 0, 0]

    # Créer une nouvelle colonne 'tempete' pour savoir si l'arbre a subit une tempete ou non
    data["tempete"] = pd.cut(data["fk_arb_etat"], bins=bins, labels=labels, right=False, ordered=False)

    return data


def print_graph(result_grid, param_grid,  param_grid1):
    results = result_grid.cv_results_
    scores_mean = results['mean_test_score']
    params = results['params']

    param_curve = param_grid[param_grid1[0]]
    param_x = param_grid[param_grid1[1]]

    reshape_score = {param: [] for param in param_curve}
    for i in range(len(params)):
        reshape_score[params[i][param_grid1[0]]].append(scores_mean[i])

    plt.figure(figsize=(8, 6))
    for param in param_curve:
        plt.plot(param_x, reshape_score[param], label=f'param_grid1[0]: {param}')
    #plt.xscale('linear')
    plt.yscale('log')
    plt.xlabel(param_grid1[1])
    plt.ylabel('Accuracy')
    plt.title('Grid Search Accuracy Results')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()


def parser_add_args(parser):
    parser.add_argument('-m', '--model', help='pretrained model name')

    parser.add_argument('-l', '--longitude', help='The longitude of the tree')
    parser.add_argument('-L', '--latitude', help='The latitude of the tree')
    parser.add_argument('-d', '--district', help='The district where the tree is planted')
    parser.add_argument('-s', '--sector', help='The sector where the tree is planted')
    parser.add_argument('-t', '--total_height', help='The height of the tree')
    parser.add_argument('-H', '--log_height', help='The height of the log')
    parser.add_argument('-R', '--diameter', help='The diameter of the log')
    parser.add_argument('-S', '--state', help='The state of the tree')
    parser.add_argument('-D', '--dev_state', help='The state of development of the tree')
    parser.add_argument('-g', '--growth_form', help='The growth form of the tree')
    parser.add_argument('-o', '--outline', help='The outline of the tree')
    parser.add_argument('-c', '--circumstances', help='The situation of the tree')
    parser.add_argument('-C', '--coating', help='If the coating is damaged')
    parser.add_argument('-a', '--age', help='The estimated age of the tree')
    parser.add_argument('-A', '--age_precision', help='The precision of the estimation of the age')
    parser.add_argument('-N', '--nb_diag', help='The number of diagnostic of the tree')
    parser.add_argument('-n', '--name', help='The technical name of the tree')
    parser.add_argument('-T', '--town', help='Who take care of the tree')
    parser.add_argument('-f', '--foliage', help='The foliage of the tree')
    parser.add_argument('-r', '--remarkable', help='If the tree is remarkable')


if __name__ == '__main__':
    data = load_data("Data_Arbre.csv", path_to_save_encoder="preprocessing/encode", encoder=None)
    normalized_data = normalize_datas(data, path_to_save="preprocessing/norm", load_file=None)
    print(normalized_data.longitude)
