import utils as ut
import argparse
import pandas as pd
import json

parser = argparse.ArgumentParser(
                    prog='agePrediction',
                    description='Predict tha age class of a tree',
                    epilog='Text at the bottom of help')
ut.parser_add_args(parser)

args = parser.parse_args()


def main(args):
    # Default model
    if not args.model:
        args.model = "1"
    # Decision Tree
    if args.model == "1":
        # Edit depending on the models
        needed = ['longitude', 'latitude', 'district', 'sector', 'total_height', 'log_height',
                  'diameter', 'state', 'dev_state', 'growth_form', 'outline', 'circumstances',
                  'coating', 'nb_diag', 'name', 'town', 'foliage', 'remarkable']
        to_keep = ["longitude", "latitude", "clc_quartier", "clc_secteur", "haut_tot", "haut_tronc", "tronc_diam",
                   "fk_arb_etat", "fk_stadedev", "fk_port", "fk_pied", "fk_situation", "fk_revetement",
                   "clc_nbr_diag", "fk_nomtech", "villeca", "feuillage", "remarquable"]
        to_encode = ["clc_quartier", "clc_secteur", "fk_arb_etat", "fk_stadedev", "fk_port", "fk_pied",
                     "fk_situation", "fk_revetement", "fk_nomtech", "villeca", "feuillage", "remarquable"]
    # KNeighbors
    elif args.model == "2":
        # Edit depending on the models
        needed = ['total_height', 'log_height', 'diameter', 'dev_state', 'coating']
        to_keep = ["haut_tot", "haut_tronc", "tronc_diam", "fk_stadedev", "fk_revetement"]
        to_encode = ["fk_stadedev", "fk_revetement"]
    # Naive Bayes
    elif args.model == "3":
        # Edit depending on the models
        needed = ['total_height', 'dev_state', 'diameter']
        to_keep = ["haut_tot", "tronc_diam", "fk_stadedev"]
        to_encode = ["fk_stadedev"]
    # Random Forest
    elif args.model == "4":
        # Edit depending on the models
        needed = ['total_height', 'dev_state', 'diameter']
        to_keep = ["haut_tot", "tronc_diam", "fk_stadedev"]
        to_encode = ["fk_stadedev"]
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
    clf = ut.load_model("../models/age" + args.model + ".pkl")
    # Print the result
    print(tree)
    res = clf.predict(tree).tolist()
    json_object = json.dumps(res, indent=4)
    # Writing to sample.json
    with open('../output/age'+args.model+'.json', "w") as outfile:
        outfile.write(json_object)
    print(res)


main(args)
