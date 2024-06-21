# Projet_IA
Projet fin d'année partie IA

## Script
### Utilisation
Entrez en argument le nom du models et les paramètres des arbres
#### Exemple
```bash
py script.py -m age3 -t 18 -R 112 -D "Adulte"
```

```bash
py script.py -m storm1 -S "EN PLACE" -L "49.840500205122" -l "3.29326360936389" -H 30 -N 0 -n "QUERUB" -T "VILLE" -f "Feuillu"
```

```bash
py script.py -m height1 -t 30 -n "QUERUB" -D "Jeune"
```

### Output
Output dans le format json dans le dossier `output` sous le nom `nomdumodel.json`

- Pour le cluster de hauteur, output un json contenant uniquement le numéro du cluster
- Pour l'âge, output un json sous la forme de tableau contenant toutes les classes d'âge prédites
- Pour l'alerte tempète, output un json sous la forme de tableau contenant les prédiction si oui (1) ou non (0) l'arbre
est suceptible de subir une tempête

### Liste des argument
```
    '-m', '--model', help : 'pretrained model name and version', type:str

    '-l', '--longitude', help : 'The longitude of the tree', type:float
    '-L', '--latitude', help : 'The latitude of the tree', type:float
    '-d', '--district', help : 'The district where the tree is planted', type:str
    '-s', '--sector', help : 'The sector where the tree is planted', type:str
    '-t', '--total_height', help : 'The height of the tree', type:int
    '-H', '--log_height', help : 'The height of the log', type:int
    '-R', '--diameter', help : 'The diameter of the log', type:int
    '-S', '--state', help : 'The state of the tree', type:str
    '-D', '--dev_state', help : 'The state of development of the tree', type:str
    '-g', '--growth_form', help : 'The growth form of the tree', type:str
    '-o', '--outline', help : 'The outline of the tree', type:str
    '-c', '--circumstances', help : 'The situation of the tree', type:str
    '-C', '--coating', help : 'If the coating is damaged', type:str
    '-a', '--age', help : 'The estimated age of the tree', type:int
    '-A', '--age_precision', help : 'The precision of the estimation of the age', type:int
    '-N', '--nb_diag', help : 'The number of diagnostic of the tree', type:int
    '-n', '--name', help : 'The technical name of the tree', type:str
    '-T', '--town', help : 'Who take care of the tree', type:str
    '-f', '--foliage', help : 'The foliage of the tree', type:str
    '-r', '--remarkable', help : 'If the tree is remarkable', type:str
```

### Modèles possible
- `height1` Clusterisation de la hauteur selon `KMeans`
- `age1` Prédiction de la classe d'âge selon `Passive Aggressive`
- `age2` Prédiction de la classe d'âge selon `Random Forest`
- `age3` Prédiction de la classe d'âge selon `SGD`
- `age4` Prédiction de la classe d'âge selon `SVM`
- `storm1` Prédiction qu'un arbre va subir une tempête selon `Decision Tree`
- `storm2` Prédiction qu'un arbre va subir une tempête selon `KNeighbors`
- `storm3` Prédiction qu'un arbre va subir une tempête selon `Naive Bayes`
- `storm4` Prédiction qu'un arbre va subir une tempête selon `Random Forest`