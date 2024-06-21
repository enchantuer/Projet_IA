import utils as ut
import argparse
import pandas as pd
import numpy as np
import json

parser = argparse.ArgumentParser(
    prog='heightCluster',
    description='Find in which height cluster the tree belongs based on pretrained models',
    epilog='Text at the bottom of help')
ut.parser_add_args(parser)


def main(args):
    # Default model
    if not args.model:
        args.model = "1"
    # Decision Tree
    if args.model == "1":
        # Edit depending on the models
        needed = ['total_height', 'name', 'dev_state']
        to_keep = ["haut_tot", "fk_nomtech", "fk_stadedev"]
        to_encode = ["fk_nomtech", "fk_stadedev"]
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
    centroids = np.loadtxt('../models/centroids'+args.model+'.csv', delimiter=',')
    # Print the result
    res = ut.get_cluster(centroids, tree[:1])
    json_object = json.dumps(res)
    # Writing to sample.json
    with open('../output/height'+args.model+'.json', "w") as outfile:
        outfile.write(json_object)
    print(res)


args = parser.parse_args()
main(args)
