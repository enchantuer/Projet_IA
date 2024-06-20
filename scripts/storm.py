import utils as ut
import argparse
import pandas as pd

parser = argparse.ArgumentParser(
                    prog='stormTreeAlert',
                    description='Predict which tree will be in danger because of the storm based on pretrained models',
                    epilog='Text at the bottom of help')
ut.parser_add_args(parser)


def main(args):
    # Default model
    if not args.model:
        args.model = "1"
    # Decision Tree
    if args.model == "1":
        # Edit depending on the models
        needed = ['longitude', 'latitude', 'log_height', 'nb_diag', 'name', 'town', 'foliage']
        to_keep = ["longitude", "latitude", "haut_tronc", "clc_nbr_diag", "fk_nomtech", "villeca", "feuillage"]
        to_encode = ["fk_nomtech", "villeca", "feuillage"]
    # KNeighbors
    elif args.model == "2":
        # Edit depending on the models
        needed = ['longitude', 'latitude', 'sector', 'dev_state', 'outline', 'nb_diag']
        to_keep = ["longitude", "latitude", "clc_secteur", "fk_stadedev", "clc_nbr_diag", "villeca"]
        to_encode = ["clc_secteur", "fk_stadedev", "fk_pied"]
    # Naive Bayes
    elif args.model == "3":
        # Edit depending on the models
        needed = ['log_height', 'dev_state', 'outline', 'coating', 'remarkable']
        to_keep = ["haut_tronc", "fk_stadedev", "fk_pied", "fk_revetement", "remarquable"]
        to_encode = ["fk_stadedev", "fk_pied", "fk_revetement", "remarquable"]
    # Random Forest
    elif args.model == "4":
        # Edit depending on the models
        needed = ['longitude', 'latitude', 'sector', 'dev_state', 'remarkable']
        to_keep = ["longitude", "latitude", "clc_secteur", "fk_stadedev", "remarquable"]
        to_encode = ["clc_secteur", "fk_stadedev", "remaquable"]
    else:
        print('Merci de précisé un model valid avec : --model \"nom_du_model\"')
        return

    # Dont edit
    stop = ut.check_missing_parameter(args, needed=needed)
    if stop:
        return
    tree = pd.DataFrame(ut.get_trees_from_parser(args), columns=to_keep)
    for col in to_encode:
        ut.encode_data(tree, col, load_file='../preprocessing/encode/encode_' + col + ".pkl")
    tree = ut.normalize_datas(tree, load_file='../preprocessing/norm')
    # Load the model
    clf = ut.load_model("../models/storm" + args.model + ".pkl")
    # Print the result
    print(clf.predict(tree))


args = parser.parse_args()
main(args)
