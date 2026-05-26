# Projet Contrôle_Apprentissage

Description
-
Ce dépôt contient plusieurs variantes d'un projet de contrôle d'un drone pour des expériences d'apprentissage (mémoire courte). Les trois dossiers principaux représentent des versions différentes du même travail : configuration d'origine et variantes utilisées pour un mémoire.

Arborescence principale
-
- `Controle_Apprentissage Origine/` : version de départ (scripts de base et modèles originaux)
- `Controle_Apprentissage Memoire 2/` : version modifiée avec mémoire sur 2 pas, scripts : `drone.py`, `multi.py`, `visualisation.py` et un dossier `model/` contenant de nombreux checkpoints `.pth`
- `Controle_Apprentissage Memoire 3/` : version modifiée avec mémoire sur 3 pas, mêmes scripts principaux et dossier `model/`

Fichiers importants
-
- `drone.py` : script principal pour exécuter/évaluer l'agent (présent dans chaque sous-dossier)
- `multi.py` : script pour exécutions multi-expérimentales / batch. Ce fichier a pour but de partir d'un model déjà existant et de l'affiner en laissant plusieurs instances venant de lui.
- `visualisation.py` : outils de visualisation des trajectoires / performances
- `model/` : dossiers contenant checkpoints PyTorch (`.pth`)

Exemples rapides
-
Se placer dans la version souhaitée puis lancer :

```bash
cd "Controle_Apprentissage Memoire 2"
python drone.py
```


Explications détaillées des scripts et fonctions
-
`drone.py` (toutes versions)
- But : simuler un drone sur une grille et entraîner un agent (DQN) pour maximiser la capture de signal depuis des "users" positionnés aléatoirement.
- Classe principale : `DroneGameAI`
	- `__init__()` : initialise la position du drone, la direction, la grille, les utilisateurs (`_place_user()`), le score initial, le timer et calcule la position optimale (`_optimal()`).
	- `_place_user()` : génère aléatoirement 1 à 5 utilisateurs (chaque user = (x, y, intensité du débit)).
	- `_score(position)` : calcule la somme des contributions de chaque utilisateur pour une position donnée (fonction inversement proportionnelle à la distance).
	- `_optimal()` : balaye la grille pour trouver la case donnant le meilleur score total (utilisée pour normaliser/évaluer la performance locale).
	- `reset()` : remet l'environnement dans un état aléatoire (nouveaux users, timer, etc.).
	- `get_state()` : renvoie l'observation fournie à l'agent (varie selon la version) :
		- Origine : `[debnorm, gradient, gradient_prev, time_norm, collision_up, collision_right, collision_down, collision_left]` (taille 8).
		- Memoire 2 : `[debnorm, gradient, gradient_prev, time_norm] + dir_actuelle(4) + dir_prev(4)` (taille 12).
		- Memoire 3 : idem Memoire 2 + `dir_prev_prev(4)` (taille 16).
	- `play_step(action, ...)` : applique une action (one-hot 4 valeurs) pour déplacer le drone, met à jour `debit`/`gradient`, calcule le `reward`, gère `done` (collision ou timer) et réinitialise les users en cas de succès proche de l'optimal.
	- `draw()` : affiche l'état si Pygame est utilisé.
	- `train()` : boucle d'entraînement qui interagit avec `Agent` : obtention d'état, sélection d'action (`agent.get_action`), entraînement court terme (`train_short_memory`) et stockage dans la mémoire pour entraînement long terme (`train_long_memory`) quand l'épisode est terminé.

Points importants sur les récompenses
- Pénalité par pas ; grosse pénalité pour collision ou temps écoulé ; bonus lors d'amélioration du `debit` (+20 + terme proportionnel au gradient) ; gros bonus quand le drone atteint ~95% du score optimal (récompense de réussite et réinitialisation des users).