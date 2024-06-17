import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

def load_data(file_path):
    # TODO : Traitement des données
    data = pd.read_csv(file_path)

    temp = OrdinalEncoder()
    train_cat = data[["fk_stadedev"]]
    a = temp.fit_transform(train_cat)
    # print(a)
    data["fk_stadedev"] = a

    return data