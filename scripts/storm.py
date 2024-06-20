from _ast import arg

import utils as ut
import argparse
import pandas as pd

nbr_model = 1

parser = argparse.ArgumentParser(
                    prog='stormTreeAlert',
                    description='Predict wich tree will be in danger because of the storm based on pretrained models',
                    epilog='Text at the bottom of help')
ut.parser_add_args(parser)

args = parser.parse_args()


def error_missing_parameter(parameters):
    print('Merci de fournir les parametres suivant :')
    for parameter in parameters:
        print('\t--', parameter)

def main(args):
    # Default model
    if not args.model:
        args.model = "1"
    # Check args
    if args.model == "1":
        needed = ['longitude', 'latitude', 'log_height', 'nb_diag', 'name', 'town', 'foliage']
    elif args.model == "2":
        needed = ['longitude', 'latitude', 'sector', 'dev_state', 'outline', 'nb_diag']
    elif args.model == "3":
        needed = ['log_height', 'dev_state', 'outline', 'coating', 'remarkable']
    elif args.model == "4":
        needed = ['longitude', 'latitude', 'sector', 'dev_state', 'remarkable']
    else:
        print('Merci de précisé un model valid avec : --model \"nom_du_model\"')
        return
    for need in needed:
        if need not in args.__dict__:
            error_missing_parameter(need)
            return
    # if not (args.longitude or args.latitude or args.sector or args.state or args.dev_state or args.remarkable):
    # Decision Tree
    if args.model == "1":
        # Edit depending on the models
        tree = pd.DataFrame({
            "longitude": [float(args.longitude)],
            "latitude": [float(args.latitude)],
            "haut_tronc": [int(args.log_height)],
            "clc_nbr_diag": [int(args.nb_diag)],
            "fk_nomtech": [args.name],
            "villeca": [args.town],
            "feuillage": [args.foliage],
        })
        # Edit depending on the models
        to_encode = ["fk_nomtech", "villeca", "feuillage"]
    # KNeighbors
    elif args.model == "2":
        # Edit depending on the models
        tree = pd.DataFrame({
            "longitude": [float(args.longitude)],
            "latitude": [float(args.latitude)],
            "clc_secteur": [args.sector],
            "fk_stadedev": [args.dev_state],
            "fk_pied": [args.outline],
            "clc_nbr_diag": [int(args.nb_diag)],
        })
        # Edit depending on the models
        to_encode = ["clc_secteur", "fk_stadedev", "fk_pied"]
    # Naive Bayes
    elif args.model == "3":
        # Edit depending on the models
        tree = pd.DataFrame({
            "haut_tronc": [int(args.log_height)],
            "fk_stadedev": [args.dev_state],
            "fk_pied": [args.outline],
            "fk_revetement": [args.coating],
            "remaquable": [args.remarkable],

        })
        # Edit depending on the models
        to_encode = ["fk_stadedev", "fk_pied", "fk_revetement", "remaquable"]
    # Random Forest
    elif args.model == "4":
        # Edit depending on the models
        tree = pd.DataFrame({
            "longitude": [float(args.longitude)],
            "latitude": [float(args.latitude)],
            "clc_secteur": [args.sector],
            "fk_stadedev": [args.dev_state],
            "remarquable": [args.remarkable],

        })
        # Edit depending on the models
        to_encode = ["clc_secteur", "fk_stadedev", "remaquable"]
    # Dont edit
    for col in to_encode:
        ut.encode_data(tree, col, load_file='../preprocessing/encode/encode_' + col + ".pkl")
    tree = ut.normalize_datas(tree, load_file='../preprocessing/norm')
    # Load the model
    clf = ut.load_model("../models/storm" + args.model + ".pkl")
    # Print the result
    print(clf.predict(tree))


main(args)
