import utils as ut
import argparse
import pandas as pd

parser = argparse.ArgumentParser(
    prog='heightCluster',
    description='Find in which height cluster the tree belongs based on pretrained models',
    epilog='Text at the bottom of help')

# Model choice
parser.add_argument('-m', '--model', help='pretrained model name')
# Input data
parser.add_argument('-t', '--height_tot', help='The total height of the tree')
parser.add_argument('-l', '--height_log', help='The total height of the log')
parser.add_argument('-n', '--tech_name', help='The technical name of the tree')
parser.add_argument('-s', '--state_dev', help='The state of development of the tree')

args = parser.parse_args()

tree = pd.DataFrame({
    "haut_tot": [int(args.height_tot)],
    "haut_tronc": [int(args.height_log)],
    "fk_nomtech": [args.tech_name],
    "fk_stadedev": [args.state_dev]
})
to_encode = ["fk_nomtech", "fk_stadedev"]
for col in to_encode:
    ut.encode_data(tree, col, load_file='../preprocessing/encode' + col + ".pkl")
tree = ut.normalize_datas(tree, load_file='../preprocessing/norm')

if not args.model:
    args.model = "1"
clf = ut.load_model("../models/height"+args.model+".pkl")
print(clf.predict(tree))
