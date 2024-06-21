import utils as ut
from sklearn.model_selection import train_test_split
import numpy as np

data = ut.load_data("Data_Arbre.csv", path_to_save_encoder="preprocessing/encode", encoder=None)
ut.normalize_datas(data, path_to_save="preprocessing/norm", load_file=None)

# Age
from age import passive_aggressive as agePA, random_forest as ageRF, SGD as ageSGD, SVM as ageSVM
# Modèle 1
d = agePA.load_data("Data_Arbre.csv")
d = d[d.haut_tot != 0]
# Séparer les caractéristiques (features) et la cible (target)
X = d.drop(columns=["age_estim", "age_class", "fk_prec_estim"])
X = ut.normalize_datas(X, load_file="preprocessing/norm")
y = d["age_class"]
# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf, grid_s, param_grid0 = agePA.get_best_model(X_train, y_train)
ut.save_model(clf, 'models/age1.pkl')
#
# Modèle 2
d = ageRF.load_data("Data_Arbre.csv")
d = d[d.haut_tot != 0]
# Séparer les caractéristiques (features) et la cible (target)
X = d.drop(columns=["age_estim", "age_class", "fk_prec_estim"])
X = ut.normalize_datas(X, load_file="preprocessing/norm")
y = d["age_class"]
# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf, grid_s, param_grid0 = ageRF.get_best_model(X_train, y_train)
ut.save_model(clf, 'models/age2.pkl')
#
# Modèle
d = ageSGD.load_data("Data_Arbre.csv")
d = d[d.haut_tot != 0]
# Séparer les caractéristiques (features) et la cible (target)
X = d.drop(columns=["age_estim", "age_class", "fk_prec_estim"])
X = ut.normalize_datas(X, load_file="preprocessing/norm")
y = d["age_class"]
# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf, grid_s, param_grid0 = ageSGD.get_best_model(X_train, y_train)
ut.save_model(clf, 'models/age3.pkl')
#
# Modèle
d = ageSVM.load_data("Data_Arbre.csv")
d = d[d.haut_tot != 0]
# Séparer les caractéristiques (features) et la cible (target)
X = d.drop(columns=["age_estim", "age_class", "fk_prec_estim"])
X = ut.normalize_datas(X, load_file="preprocessing/norm")
y = d["age_class"]
# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf, grid_s, param_grid0 = ageSVM.get_best_model(X_train, y_train)
ut.save_model(clf, 'models/age4.pkl')


# Height
from height import KMeans as heightKM
# Modèle 1
d = heightKM.load_data("Data_Arbre.csv")
d = d[d.haut_tot != 0]
# Remove coordinate for the model
X = d.drop(columns=["longitude", "latitude"])
# Normalize the data
X = ut.normalize_datas(X, load_file="preprocessing/norm")
# Generate the model
m = heightKM.generate_model(X, 2)
np.savetxt('models/centroids1.csv', m.cluster_centers_, delimiter=',')


# Storm
from storm import decision_tree as stormDT, KNeighbors as stormKN, naive_bayes as stormNB, random_forest as stormRF
# Modèle 1
d = stormDT.load_data("Data_Arbre.csv")
# Séparer les caractéristiques (features) et la cible (target)
X = d.drop(columns=["fk_arb_etat", "tempete"])
y = d["tempete"]
# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf, grid_s, param_grid0 = stormDT.get_best_model(X_train, y_train)
ut.save_model(clf, 'models/storm1.pkl')
#
# Modèle 2
d = stormKN.load_data("Data_Arbre.csv")
# Séparer les caractéristiques (features) et la cible (target)
X = d.drop(columns=["fk_arb_etat", "tempete"])
y = d["tempete"]
# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf, grid_s, param_grid0 = stormKN.get_best_model(X_train, y_train)
ut.save_model(clf, 'models/storm2.pkl')
#
# Modèle 3
d = stormNB.load_data("Data_Arbre.csv")
# Séparer les caractéristiques (features) et la cible (target)
X = d.drop(columns=["fk_arb_etat", "tempete"])
y = d["tempete"]
# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf, grid_s, param_grid0 = stormNB.get_best_model(X_train, y_train)
ut.save_model(clf, 'models/storm3.pkl')
#
# Modèle 4
d = stormRF.load_data("Data_Arbre.csv")
# Séparer les caractéristiques (features) et la cible (target)
X = d.drop(columns=["fk_arb_etat", "tempete"])
y = d["tempete"]
# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf, grid_s, param_grid0 = stormRF.get_best_model(X_train, y_train)
ut.save_model(clf, 'models/storm4.pkl')
