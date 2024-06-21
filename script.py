import argparse
import pandas as pd
import json
import utils as ut
import numpy as np


parser = argparse.ArgumentParser(
                    prog='models',
                    description='Predict tha age class, height cluster, storm alert of a tree',)
ut.parser_add_args(parser)


def main(args):
    # Passive Aggressive
    if args.model == "age1":
        # Edit depending on the models
        needed = ['longitude', 'latitude', 'district', 'sector', 'total_height', 'log_height',
                  'diameter', 'state', 'dev_state', 'growth_form', 'outline', 'circumstances',
                  'coating', 'nb_diag', 'name', 'town', 'foliage', 'remarkable']
        to_keep = ["longitude", "latitude", "clc_quartier", "clc_secteur", "haut_tot", "haut_tronc", "tronc_diam",
                   "fk_arb_etat", "fk_stadedev", "fk_port", "fk_pied", "fk_situation", "fk_revetement",
                   "clc_nbr_diag", "fk_nomtech", "villeca", "feuillage", "remarquable"]
        to_encode = ["clc_quartier", "clc_secteur", "fk_arb_etat", "fk_stadedev", "fk_port", "fk_pied",
                     "fk_situation", "fk_revetement", "fk_nomtech", "villeca", "feuillage", "remarquable"]
    # Random Forest
    elif args.model == "age2":
        # Edit depending on the models
        needed = ['total_height', 'log_height', 'diameter', 'dev_state', 'coating']
        to_keep = ["haut_tot", "haut_tronc", "tronc_diam", "fk_stadedev", "fk_revetement"]
        to_encode = ["fk_stadedev", "fk_revetement"]
    # SGD
    elif args.model == "age3":
        # Edit depending on the models
        needed = ['total_height', 'dev_state', 'diameter']
        to_keep = ["haut_tot", "tronc_diam", "fk_stadedev"]
        to_encode = ["fk_stadedev"]
    # SVM
    elif args.model == "age4":
        # Edit depending on the models
        needed = ['total_height', 'dev_state', 'diameter']
        to_keep = ["haut_tot", "tronc_diam", "fk_stadedev"]
        to_encode = ["fk_stadedev"]
    # KMeans
    elif args.model == "height1":
        # Edit depending on the models
        needed = ['total_height', 'name', 'dev_state']
        to_keep = ["haut_tot", "fk_nomtech", "fk_stadedev"]
        to_encode = ["fk_nomtech", "fk_stadedev"]
    # Decision Tree
    elif args.model == "storm1":
        # Edit depending on the models
        needed = ['longitude', 'latitude', 'log_height', 'nb_diag', 'name', 'town', 'foliage']
        to_keep = ["longitude", "latitude", "haut_tronc", "clc_nbr_diag", "fk_nomtech", "villeca", "feuillage"]
        to_encode = ["fk_nomtech", "villeca", "feuillage"]
    # KNeighbors
    elif args.model == "storm2":
        # Edit depending on the models
        needed = ['longitude', 'latitude', 'sector', 'dev_state', 'outline', 'nb_diag']
        to_keep = ["longitude", "latitude", "clc_secteur", "fk_stadedev", "clc_nbr_diag", "villeca"]
        to_encode = ["clc_secteur", "fk_stadedev", "fk_pied"]
    # Naive Bayes
    elif args.model == "storm3":
        # Edit depending on the models
        needed = ['log_height', 'dev_state', 'outline', 'coating', 'remarkable']
        to_keep = ["haut_tronc", "fk_stadedev", "fk_pied", "fk_revetement", "remarquable"]
        to_encode = ["fk_stadedev", "fk_pied", "fk_revetement", "remarquable"]
    # Random Forest
    elif args.model == "storm4":
        # Edit depending on the models
        needed = ['longitude', 'latitude', 'sector', 'dev_state', 'remarkable']
        to_keep = ["longitude", "latitude", "clc_secteur", "fk_stadedev", "remarquable"]
        to_encode = ["clc_secteur", "fk_stadedev", "remaquable"]
    else:
        print('Merci de précisé un model valid avec : --model \"nom_du_model\"')
        possible_models = ['height1', 'age1', 'age2', 'age3', 'age4', 'storm1', 'storm2', 'storm3', 'storm4']
        print ('Modèles possibles : ')
        for model in possible_models:
            print('\t', model)
        return

    # Dont edit
    stop = ut.check_missing_parameter(args, needed=needed)
    if stop:
        return
    tree = pd.DataFrame(ut.get_trees_from_parser(args), columns=to_keep)
    for col in to_encode:
        ut.encode_data(tree, col, load_file='preprocessing/encode/encode_' + col + ".pkl")
    tree = ut.normalize_datas(tree, load_file='preprocessing/norm')
    # Load the model
    if args.model == "height1":
        # Load the model
        centroids = np.loadtxt('models/centroids1.csv', delimiter=',')
        # Print the result
        res = ut.get_cluster(centroids, tree[:1])
    else:
        clf = ut.load_model("models/" + args.model + ".pkl")
        res = clf.predict(tree).tolist()
    # Output the result json
    json_object = json.dumps(res, indent=4)
    with open('output/'+args.model+'.json', "w") as outfile:
        outfile.write(json_object)
    # Log the result
    print(res)


args = parser.parse_args()
main(args)
