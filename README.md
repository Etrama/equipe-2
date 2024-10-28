# 🏭 Application de Prédiction pour Procédé de Flottation Minière

Ce projet est un point de départ pour le hackaton de prédiction du taux de silice dans le processus de flottation minière. Il fournit une base d'application avec dashboard et module de prédiction utilisant Streamlit.

## 🎯 Objectifs du Hackaton

- Développer un modèle de régression pour prédire le taux de silice
- Créer une interface de visualisation interactive des données
- Déployer l'application sur Azure
- Présenter la solution finale

## 🚀 Démarrage Rapide

### Prérequis
```bash
Python 3.8+
pip install -r requirements.txt
```

### Installation
1. Cloner le repository
2. Installer les dépendances :
```bash
pip install numpy pandas plotly streamlit scikit-learn
```
3. Lancer l'application :
```bash
streamlit run app.py

# ou
python -m streamlit run app.py
```

## 📂 Accès aux Données

TODOOO

## 📊 Structure de l'Application

### Dashboard exemple (Page 1)
- Visualisation de l'évolution temporelle de la variable cible
- Statistiques descriptives
- Paramètres de lissage ajustables
- Métriques clés en temps réel

### Prédiction exemple (Page 2)
- Interface de saisie des variables du procédé
- Prédiction en temps réel
- Affichage des résultats
- Résumé des valeurs d'entrée

## 💻 Structure du Code

```
.
├── app.py              # Application principale
├── requirements.txt    # Dépendances
└── README.md          # Documentation
```

## 🔧 Fonctionnalités à Implémenter

1. **Modélisation**
   - Adapter le modèle aux données réelles de flottation
   - Optimiser les hyperparamètres
   - Implémenter la validation croisée

2. **Dashboard**
   - Ajouter des visualisations spécifiques au procédé
   - Intégrer des alertes pour les valeurs critiques
   - Créer des vues personnalisées

3. **Déploiement**
   - Configurer le déploiement sur Azure
   - Mettre en place un pipeline CI/CD
   - Gérer la mise à jour des prédictions

## 👥 Organisation de l'Équipe

- **Data Scientist** : Modélisation et optimisation
- **Data Analyst** : Création du dashboard et visualisations
- **Data Engineer** : Déploiement et infrastructure

## ⏰ Planning du Hackaton

- **11h-11h30** : Setup et organisation
- **11h30-16h** : Développement et implémentation
- **16-17** : Présentation des solutions

## 📝 Critères d'Évaluation

- Qualité du modèle de prédiction
- Ergonomie et clarté du dashboard
- Robustesse du déploiement
- Qualité de la présentation finale

## 🤝 Contribution

1. Cloner le projet
2. Créer une branche par équipe (`git checkout -b feature/AmazingFeature`)
3. Commit & push les changements
4. Créer une merge request à la fin du temps imparti

## 🔗 Ressources Utiles

- Documentation Streamlit : [https://docs.streamlit.io/]
- Azure Deployment Guide : [https://docs.microsoft.com/azure]