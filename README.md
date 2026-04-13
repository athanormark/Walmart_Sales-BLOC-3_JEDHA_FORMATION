# Walmart - Prédiction des ventes hebdomadaires

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=fff)](#)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=fff)](#)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=fff)](#)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=fff)](#)
[![scikit--learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=fff)](#)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat)](#)
[![Seaborn](https://img.shields.io/badge/Seaborn-444876?style=flat)](#)
[![JEDHA](https://img.shields.io/badge/JEDHA-blueviolet?style=flat)](#)

---

## About

Le service marketing de **Walmart** souhaite un modèle de Machine Learning capable d'estimer les ventes hebdomadaires de ses magasins avec la meilleure précision possible. L'enjeu est double :

* Comprendre comment les ventes sont influencées par les indicateurs économiques (chômage, prix du carburant, indice des prix à la consommation) et par la saisonnalité (périodes de fêtes, mois de l'année).
* Planifier les futures campagnes marketing en identifiant les magasins, les périodes et les leviers sur lesquels concentrer les investissements.

La variable cible est `Weekly_Sales` (ventes hebdomadaires en dollars). Le modèle doit fournir des prédictions fiables et des coefficients interprétables pour orienter les décisions stratégiques.

Projet réalisé dans le cadre du **BLOC 3 -- Machine Learning** de la formation Data Fullstack (JEDHA Bootcamp).

---

## Dataset

* **Source** : dataset custom fourni par JEDHA
* **Fichier** : `Walmart_Store_sales.csv`
* **Dimensions** : 150 lignes, 8 colonnes
* **Après nettoyage** (target manquante + outliers 3 sigmas) : 131 lignes exploitables

|Colonne|Description|
|-|-|
|Store|Identifiant du magasin|
|Date|Semaine de vente|
|Weekly_Sales|Ventes hebdomadaires en dollars (target)|
|Holiday_Flag|Semaine fériée (1) ou non (0)|
|Temperature|Température moyenne de la région (Fahrenheit)|
|Fuel_Price|Prix du carburant dans la région|
|CPI|Indice des prix à la consommation|
|Unemployment|Taux de chômage dans la région|

---

## Installation

```bash
git clone https://github.com/athanormark/Walmart_Sales-BLOC-3_JEDHA_FORMATION.git
cd Walmart_Sales-BLOC-3_JEDHA_FORMATION
pip install -r requirements.txt
```

Placer `Walmart_Store_sales.csv` dans `data/raw/`, puis :

```bash
jupyter notebook notebook/01_eda_and_baseline.ipynb
```

---

## Pipeline

### 1. EDA et nettoyage

* **Target manquante** : 14 lignes supprimées (pas d'imputation sur la target pour éviter le biais)
* **Feature engineering dates** : extraction de Year, Month, Day, DayOfWeek depuis la colonne `Date`
* **Outliers** : règle des 3 sigmas appliquée sur Temperature, Fuel_Price, CPI, Unemployment (5 lignes supprimées)
* **Dataset final** : 131 lignes, 10 features

### 2. Preprocessing (scikit-learn)

|Étape|Méthode|Justification|
|-|-|-|
|Split|`train_test_split(test_size=0.2, random_state=42)`|80/20, seed fixe pour reproductibilité|
|Numériques|`SimpleImputer(mean)` + `StandardScaler`|Imputation des NaN restants + mise à l'échelle|
|Catégorielles|`SimpleImputer(most_frequent)` + `OneHotEncoder(drop='first')`|`drop='first'` évite la multicolinéarité|
|Assemblage|`ColumnTransformer` dans un `Pipeline`|`fit` sur train uniquement pour éviter le data leakage|

### 3. Modélisation

1. **Régression Linéaire** (baseline) : minimisation OLS, point de référence
2. **Ridge (alpha=100)** : test avec alpha élevé pour illustrer l'impact d'une pénalisation excessive
3. **Ridge optimisé** (`GridSearchCV`, cv=5) : recherche du meilleur alpha dans [0.01, 0.1, 1, 10, 50, 100, 500, 1000]
4. **Lasso optimisé** (`GridSearchCV`, cv=5) : régularisation L1 avec sélection de features automatique

---

## Résultats

### Comparaison des modèles

|Modèle|R2 Train|R2 Test|MAE Test|RMSE Test|
|-|-|-|-|-|
|Régression Linéaire|0.977|0.891|153 208 $|194 682 $|
|Ridge optimisé (alpha=0.01)|0.977|0.892|151 995 $|193 651 $|
|**Lasso optimisé (alpha=500)**|**0.977**|**0.897**|**151 358 $**|**188 738 $**|

**Constats** :

* Le modèle explique environ 90% de la variance des ventes sur le jeu de test.
* L'écart train/test (0.977 vs 0.891) indique un léger overfitting, corrigé partiellement par la régularisation.
* Le best alpha Ridge très faible (0.01) confirme que l'overfitting initial était modéré.
* Lasso (alpha=500) obtient le meilleur R2 test (0.897) et élimine 1 feature sur 27, ce qui simplifie le modèle.

### Interprétation des coefficients

L'analyse des coefficients révèle la hiérarchie suivante :

* **Store** (identité du magasin) : de loin le facteur le plus prédictif. Chaque magasin a un niveau de ventes propre, lié à sa taille, sa localisation et sa zone de chalandise.
* **Month / DayOfWeek** : la saisonnalité joue un rôle significatif. Les ventes augmentent nettement en novembre-décembre (fêtes de fin d'année et Black Friday).
* **Holiday_Flag** : les semaines fériées montrent un impact positif modéré sur les ventes.
* **Temperature, Fuel_Price, CPI, Unemployment** : ces indicateurs économiques montrent une corrélation linéaire faible avec les ventes. Leur pouvoir prédictif est marginal dans ce modèle.

---

## Conclusion

Le projet répond à la problématique posée par le service marketing de Walmart : **prédire les ventes hebdomadaires pour orienter les stratégies marketing et de stockage**.

### Performance du modèle

Le modèle **Lasso (alpha=500)** obtient le meilleur compromis : R2 test = 0.897, MAE = 151 358 $, RMSE = 188 738 $. La régularisation L1 simplifie le modèle en éliminant 1 feature sur 27, sans perte de performance.

### Recommandations pour Walmart

* Adapter les budgets marketing par magasin : l'identité du magasin est le premier déterminant des ventes. Les campagnes doivent être calibrées magasin par magasin, et non de manière uniforme sur l'ensemble du réseau.
* Concentrer les investissements sur novembre-décembre : la saisonnalité est le deuxième levier le plus puissant. Les pics de fin d'année représentent une opportunité majeure pour intensifier les campagnes promotionnelles et ajuster les stocks.
* Exploiter les semaines fériées : le Holiday_Flag montre un impact positif sur les ventes. Les périodes fériées justifient des opérations commerciales spécifiques.
* Ne pas surestimer les indicateurs macroéconomiques : Temperature, Fuel_Price, CPI et Unemployment ont un pouvoir prédictif marginal. Le service marketing ne devrait pas conditionner ses campagnes à ces variables.

### Limites

* Le dataset est très petit (131 lignes après nettoyage), ce qui limite la généralisation.
* Le OneHotEncoding de Store crée 44 colonnes pour seulement 131 observations.
* Un dataset plus large et un modèle non linéaire (Random Forest, XGBoost) pourraient capter des interactions entre variables.

---

## Structure du projet

```text
walmart-sales-prediction/
├── data/
│   └── raw/                  # Données brutes (non versionné)
├── notebook/
│   └── 01_eda_and_baseline.ipynb
├── assets/
│   └── images/               # Graphiques exportés du notebook
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Auteur

Athanor SAVOUILLAN · [GitHub](https://github.com/athanormark)

