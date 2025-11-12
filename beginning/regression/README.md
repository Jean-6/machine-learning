
🎯 Objectif

Créer un modèle complet de régression linéaire from scratch, sans librairies ML.
A coder :

- génération d’un dataset
- nettoyage / normalisation
- implémentation de la fonction de coût
- implémentation du gradient
- boucle d’entraînement (gradient descent)
- prédictions
- visualisations
- évaluation finale


🧩 Contexte : Prédire le prix d’une maison

Une agence immobilière te demande de créer un mini-modèle capable d’estimer le prix d’un bien immobilier à partir de 3 caractéristiques :

1. surface (m²)
2. nb_chambres
3. distance_centre (km du centre-ville)


Coder un modèle qui apprend la formule :

price = w1 * surface + w2 * nb_chambres + w3 * distance_centre + b

Partie 1 — Génération d’un dataset

Coder une fonction Python qui génère 500 maisons avec :

1. surface : entre 20 et 200 m²
2. chambres : entre 1 et 6
3. distance : entre 0 km et 30 km

prix défini par une formule réaliste :

prix = 2500 * surface + 15000 * nb_chambres - 2000 * distance_centre + bruit aléatoire

où le bruit est un bruit gaussien (ex : np.random.normal(0, 20000)).

Tu obtiens X (features) et y (prix).


Partie 2 — Normalisation des données

Coder :

une fonction normalize(X) qui retourne :

1. les données normalisées
2. les moyennes
3. les écarts-types

une fonction normalize_with_stats(X, mean, std) à utiliser pour la prédiction

### Standardisation

la fonction normalize_with_stats(X, mean, std) est une étape clé dans tout pipeline de Machine Learning, en particulier pour la régression linéaire, car elle garantit que les variables d’entrée (les features) ont des échelles comparables.