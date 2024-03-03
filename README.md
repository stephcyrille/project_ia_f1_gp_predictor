# Projet ML Avancé : Prediction des résultats de course de F1

## Description

Ce projet à pour but de mettre en place un outil de prédiction de position d'un pilote pilote dans un Grand Prix de F1. Les données necessaires à la réalisation de cet outils sont:
- Les informations du Circuit
- Les informations de constructeurs
- Les informations de pilotes
- Le temps en course
- Les informations de résultat des courses
- Les informations sur la saison

Les données sont collectée à partir de 1950, date du premier Grand Prix, jusqu'en 2023. Ces données ont été collectée sur <a href='https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020?select=races.csv' target='_blank' title='Accéder au site'>Kaggle</a> où vous trouverez également la documentation relative au jeu de donné.

## Structure du projet
    ├── data/                           # Données brutes
    │   ├── inputs/                     # Données brutes d'entrée (csv)
    │   └── outputs/                    # Données brutes transformées
    ├── notebooks/                      # Repertoire Jupyter notebooks 
    │   └── analysis/                   # Notebooks analyse
    │   └── data_prep/                  # Notebooks préparation des données
    │   └── modelisation/               # Notebooks de modélisation
    │       ├── supervised              # Modélisation supervisée      
    │       └── unsupervised            # Modélisation non supervisée        
    ├── src/                            # Répertoire des code modeules personnels 
    ├── requirements.txt                # Dépendances du projet
    └── README.md                       # Project README file


## Etapes de traitements
- ### Etape 1: La Data preparation
  Dans cette étapes l'on procèdes a la lecture, traitement des différents Datasets puis à leur fusion en un jeu de données comportant toutes les carractéristiques necessaire au traitements suivants.<br>
  Tous les notebooks de cette étape sont dans le dossier ```notebooks/data_prep/```

- ### Etape 2: Analyse:
  Tous les notebooks de cette étape sont dans le dossier ```notebooks/analysis/```

- ### Etape 3: Modélisation non supervisée:
  Tous les notebooks de cette étape sont dans le dossier ```notebooks/modelisation/unsuppervised/```

- ### Etape 4: Modélisation supervisée et amélioration du modèle
    Tous les notebooks de cette étape sont dans le dossier ```notebooks/modelisation/supervised/```