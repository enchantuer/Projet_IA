import utils as ut
import argparse
import pandas as pd

nbr_model = 1

parser = argparse.ArgumentParser(
                    prog='stormTreeAlert',
                    description='Predict wich tree will be in danger because of the storm based on pretrained models',
                    epilog='Text at the bottom of help')

parser.add_argument('-m', '--model', help='pretrained model name')
parser.add_argument('-ae', '--arb_etat', help='The state of the tree')
parser.add_argument('-l', '--longitude', help='The longitude of the tree')
parser.add_argument('-L', '--latitude', help='The latitude of the tree')
parser.add_argument('-ht', '--height_tronc', help='The height of the log')
parser.add_argument('-nd', '--nbr_diag', help='The number of diagnostic of the tree')
parser.add_argument('-cnd', '--clc_nbr_diag', help='The number of diagnostic of the tree')
parser.add_argument('-n', '--tech_name', help='The technical name of the tree')
parser.add_argument('-t', '--town', help='Who take care of the tree')
parser.add_argument('-f', '--foliage', help='The foliage of the tree')
parser.add_argument('-s', '--sector', help='The sector where the tree is planted')
parser.add_argument('-s', '--sector', help='The sector where the tree is planted')
parser.add_argument('-s', '--state_dev', help='The state of development of the tree')
parser.add_argument('-ft', '--foot', help='???')
parser.add_argument('-c', '--coating', help='If the coating is damaged')
parser.add_argument('-pa', '--precision_age', help='The precision of the estimage age')
parser.add_argument('-r', '--remarkable', help='If the tree is remarkable')


if nbr_model == 1:
    clf = ut.load_model("../models/storm1.pkl")
elif nbr_model == 2:
    clf = ut.load_model("../models/storm2.pkl")
elif nbr_model == 3:
    clf = ut.load_model("../models/storm3.pkl")
else:
    clf = ut.load_model("../models/storm4.pkl")


norm = ut.load_model("../models/norm.pkl")

args = parser.parse_args()
