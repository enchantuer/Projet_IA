import utils as ut
import argparse
import pandas as pd

parser = argparse.ArgumentParser(
                    prog='heightCluster',
                    description='Find in which height cluster the tree belongs based on pretrained models',
                    epilog='Text at the bottom of help')

parser.add_argument('-m', '--model', help='pretrained model name')
parser.add_argument('-t', '--height_tot', help='The total height of the tree')
parser.add_argument('-l', '--height_log', help='The total height of the log')
parser.add_argument('-n', '--tech_name', help='The technical name of the tree')
parser.add_argument('-s', '--sate_dev', help='The sate of development of the tree')

clf = ut.load_model("../models/height.pkl")
norm = ut.load_model("../models/norm.pkl")

args = parser.parse_args()
tree = pd.DataFrame({0:int(args.height_tot), 1:int(args.height_log), 2:args.tech_name, 3:args.sate_dev}, columns=[
                            "haut_tot",
                            "haut_tronc",
                            "fk_nomtech",
                            "fk_stadedev",
                        ])
liste_modif = ["clc_quartier", "clc_secteur", "fk_arb_etat", "fk_stadedev", "fk_port", "fk_pied",
               "fk_situation", "fk_revetement", "fk_nomtech", "villeca", "feuillage", "remarquable"]
normalised = norm.transform(tree)
pd.DataFrame(normalised, columns=[col for col in tree.columns if col in liste_modif] + [col for col in tree.columns if col not in liste_modif])

