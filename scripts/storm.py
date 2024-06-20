import utils as ut
import argparse
import pandas as pd

nbr_model = 1

parser = argparse.ArgumentParser(
                    prog='stormTreeAlert',
                    description='Predict wich tree will be in danger because of the storm based on pretrained models',
                    epilog='Text at the bottom of help')

parser.add_argument('-m', '--model', help='pretrained model name')

parser.add_argument('-a', '--arb_state', help='The state of the tree')
parser.add_argument('-l', '--longitude', help='The longitude of the tree')
parser.add_argument('-L', '--latitude', help='The latitude of the tree')
parser.add_argument('-h', '--height_log', help='The height of the log')
parser.add_argument('-n', '--nbr_diag', help='The number of diagnostic of the tree')
parser.add_argument('-c', '--clc_nbr_diag', help='The number of diagnostic of the tree')
parser.add_argument('-n', '--tech_name', help='The technical name of the tree')
parser.add_argument('-t', '--town', help='Who take care of the tree')
parser.add_argument('-f', '--foliage', help='The foliage of the tree')
parser.add_argument('-s', '--sector', help='The sector where the tree is planted')
parser.add_argument('-S', '--state_dev', help='The state of development of the tree')
parser.add_argument('-F', '--foot', help='???')
parser.add_argument('-c', '--coating', help='If the coating is damaged')
parser.add_argument('-p', '--precision_age', help='The precision of the estimation of the age')
parser.add_argument('-r', '--remarkable', help='If the tree is remarkable')

args = parser.parse_args()

# Edit depending on the models
tree = pd.DataFrame({
    "haut_tot": [int(args.height_tot)],
    "haut_tronc": [int(args.height_log)],
    "fk_nomtech": [args.tech_name],
    "fk_stadedev": [args.state_dev]
})
# Edit depending on the models
to_normalise = ["fk_nomtech", "fk_stadedev"]
# Dont edit
for col in to_normalise:
    ut.encode_data(tree, col, load_file='../preprocessing/' + col + ".pkl")
# Default model
if not args.model:
    args.model = "1"
# Load the model
clf = ut.load_model("../models/storm"+args.model+".pkl")
# Print the result
print(clf.predict(tree[0]))