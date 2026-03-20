# Walmart - Prediction des ventes hebdomadaires

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=flat&logo=jupyter)](https://jupyter.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data_Analysis-150458?style=flat&logo=pandas)](https://pandas.pydata.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-f7931e?style=flat&logo=scikit-learn)](https://scikit-learn.org/)

## About

Le service marketing de **Walmart** souhaite un modele de Machine Learning capable d'estimer les ventes hebdomadaires de ses magasins a partir d'indicateurs economiques (chomage, prix du carburant, CPI) et de donnees saisonnieres.

L'objectif est de predire `Weekly_Sales` avec la meilleure precision possible pour orienter les futures campagnes marketing.

Projet realise dans le cadre du **BLOC 3** de la formation Data Analyst & IA (JEDHA Bootcamp).

## Dataset

- **Source** : dataset custom fourni par JEDHA (issu d'un challenge Kaggle modifie)
- **Fichier** : `Walmart_Store_sales.csv`
- **Dimensions** : 150 lignes, 8 colonnes
- **Apres nettoyage** (target manquante + outliers 3 sigmas) : 131 lignes exploitables

| Colonne | Description |
|---------|-------------|
| Store | Identifiant du magasin |
| Date | Semaine de vente |
| Weekly_Sales | Ventes hebdomadaires (target) |
| Holiday_Flag | Semaine feriee (1) ou non (0) |
| Temperature | Temperature moyenne de la region |
| Fuel_Price | Prix du carburant dans la region |
| CPI | Indice des prix a la consommation |
| Unemployment | Taux de chomage dans la region |

## Installation

```bash
git clone https://github.com/athanormark/Walmart_Sales-BLOC-3_JEDHA_FORMATION.git
cd Walmart_Sales-BLOC-3_JEDHA_FORMATION
pip install -r requirements.txt
```

Placer `Walmart_Store_sales.csv` dans `data/raw/`, puis :

```bash
jupyter notebook notebooks/01_eda_and_baseline.ipynb
```

## Pipeline

### 1. EDA et nettoyage

- **Target manquante** : 14 lignes supprimees (pas d'imputation sur la target pour eviter le biais)
- **Feature engineering dates** : extraction de Year, Month, Day, DayOfWeek depuis la colonne `Date`
- **Outliers** : regle des 3 sigmas sur Temperature, Fuel_Price, CPI, Unemployment (5 lignes supprimees)
- **Dataset final** : 131 lignes, 10 features

### 2. Preprocessing (Scikit-Learn)

| Etape | Methode | Justification |
|-------|---------|---------------|
| Split | `train_test_split(test_size=0.2, random_state=42)` | 80/20, seed fixe pour reproductibilite |
| Numeriques | `SimpleImputer(mean)` + `StandardScaler` | Imputation des NaN restants + mise a l'echelle |
| Categorielles | `SimpleImputer(most_frequent)` + `OneHotEncoder(drop='first')` | `drop='first'` evite la multicolinearite |
| Pipeline | `ColumnTransformer` | `fit` sur train uniquement pour eviter le data leakage |

### 3. Modelisation

1. **Regression Lineaire** (baseline) : minimisation OLS
2. **Ridge (alpha=100)** : test volontaire avec alpha eleve pour montrer l'impact d'une regularisation trop forte
3. **Ridge optimise (GridSearchCV, cv=5)** : recherche du meilleur alpha
4. **Lasso optimise (GridSearchCV, cv=5)** : regularisation L1 avec selection de features

## Resultats

### Comparaison des modeles

| Modele | R2 Train | R2 Test | MAE Test | RMSE Test |
|--------|----------|---------|----------|-----------|
| Regression Lineaire | 0.977 | 0.891 | 153 208 $ | 194 682 $ |
| Ridge optimise (alpha=0.01) | 0.977 | 0.892 | 151 995 $ | 193 651 $ |
| **Lasso optimise (alpha=500)** | **0.977** | **0.897** | **151 358 $** | **188 738 $** |

**Constats** :
- Le modele explique environ 90 % de la variance des ventes sur le jeu de test
- L'ecart train/test (0.977 vs 0.891) indique un leger overfitting, corrige partiellement par la regularisation
- Le best alpha Ridge tres faible (0.01) confirme que le modele lineaire n'etait pas tres overfit
- Lasso (alpha=500) donne le meilleur R2 test (0.897) et elimine 1 feature sur 27
- Le Store (identite du magasin) est de loin le facteur le plus predictif

### Visualisations

| Saisonnalite des ventes | Feature Importance |
|:-:|:-:|
| ![Saisonnalite](assets/images/seasonality.png) | ![Feature Importance](assets/images/feature_importance.png) |

| Distribution de la target | Reel vs Predit |
|:-:|:-:|
| ![Distribution](assets/images/distribution_target.png) | ![Real vs Predicted](assets/images/real_vs_predicted.png) |

## Structure du projet

```
walmart-sales-prediction/
├── data/
│   └── raw/                  # Donnees brutes (non versionne)
├── notebooks/
│   └── 01_eda_and_baseline.ipynb
├── assets/
│   └── images/               # Graphiques exportes du notebook
├── .gitignore
├── requirements.txt
└── README.md
```

## Auteur

Athanor SAVOUILLAN · [GitHub](https://github.com/athanormark)
