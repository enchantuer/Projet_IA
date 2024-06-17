import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

def load_data(file_path):
    data = pd.read_csv(file_path)

    liste_modif=["clc_quartier","clc_secteur","fk_arb_etat","fk_stadedev","fk_port","fk_pied","fk_situation","fk_revetement","fk_nomtech","villeca","feuillage","remarquable"]
    for i in liste_modif:
        modif_data(data,i)
    return data


def modif_data(data,column):
    temp = OrdinalEncoder()
    train_cat = data[[column]]
    data[column] = temp.fit_transform(train_cat)

    return data


if __name__ == "__main__":
    print(load_data("Data_Arbre.csv"))
    pass
