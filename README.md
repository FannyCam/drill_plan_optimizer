# Drill Plan Optimizer 

**Optimisation du plan de forage**

Ce projet calcule et optimise une maille de forage (grille de trous) à l'intérieur d'un polygone défini, en respectant les contraintes de sécurité (bootlegs).

## Installation
- Créer l'environnement virtuel : `python -m venv venv`
- Activer l'environnement virtuel : `.\venv\Scripts\activate`
- Installer les dépendances : `pip install -r requirements.txt`

## Utilisation
Le programme principal se lance via la commande python `src/main.py`

Bouton 'Load JSON' : Permet de charger le fichier d'input contenant les points de bordure (boundary) et les points bootlegs.

Bouton 'Run Optimization' : Permet de lancer l'optimisation. 
Note : L'algorithme étant intensif, la fenêtre peut temporairement "freezer" pendant le calcul.

On peut voir l'avancement de l'algorithme dans le fichier data\algo.log.
Une fois l'optimisation terminée, le fichier data/holes.json sera créé et la solution sera affiché dans la fenêtre.

Il est possible de relancer l'optimisation pour améliorer la solution affichée si elle n'est pas assez bonne.

## Structure des données
- Input : Points boundary (polygone) et bootlegs (zones d'exclusion)
- Output : Grille de forage optimisée (holes.json)
