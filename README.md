# Prévision de la Demande de Vélos en Libre-Service à Londres (Régression & Séries Temporelles)

Ce projet implémente une chaîne complète de machine learning supervisé visant à modéliser et prédire la demande horaire de vélos en libre-service à Londres (*TfL cycle hire*), à partir de facteurs météorologiques, calendaires et temporels.

L'objectif métier consiste à anticiper les tensions de flux pour optimiser la logistique de redistribution des flottes et la planification opérationnelle.

---

## 1. Données & Problématique Métier

* **Jeu de données** : 17 414 observations horaires combinant les volumes de location réels et les conditions atmosphériques.
* **Variable cible (`cnt`)** : Nombre total de vélos loués par heure (moyenne : 1 143 vélos/heure, max : 7 860).
* **Variables explicatives** :
  * Météo : Température réelle (`t1`), humidité (`hum`), vitesse du vent (`wind_speed`), code météo (`weather_code`).
  * Calendrier : Jour férié (`is_holiday`), week-end (`is_weekend`), saison (`season`).
  * Temporel : Horodatage complet (`timestamp`).

---

## 2. Démarche Préparatoire & Feature Engineering

1. **Traitement de la colinéarité** :
   * Une corrélation quasi unitaire a été identifiée entre la température réelle (`t1`) et la température ressentie (`t2`).
   * La variable `t2` a été supprimée pour éliminer la redondance et stabiliser l'apprentissage des coefficients linéaires.
2. **Encodage cyclique des composantes temporelles** :
   * L'heure de la journée (0 à 23h) et le mois de l'année (1 à 12) sont des grandeurs périodiques (23h est contigu à 00h).
   * Transformation trigonométrique appliquée :
     $$\text{sin\_hour} = \sin\left(\frac{2\pi \cdot \text{hour}}{24}\right), \quad \text{cos\_hour} = \cos\left(\frac{2\pi \cdot \text{hour}}{24}\right)$$
     $$\text{sin\_month} = \sin\left(\frac{2\pi \cdot \text{month}}{12}\right), \quad \text{cos\_month} = \cos\left(\frac{2\pi \cdot \text{month}}{12}\right)$$
3. **Stratégie de découpage chronologique** :
   * Conformément aux contraintes des séries temporelles, les données ont été scindées chronologiquement (80 % train, 20 % test) sans mélange aléatoire, prévenant toute fuite d'information future (*look-ahead bias*).
4. **Pipeline de pré-traitement** :
   * Imputation médiane (`SimpleImputer`) et standardisation (`StandardScaler`) encapsulées via `ColumnTransformer`.

---

## 3. Comparaison des Modèles & Résultats

Quatre architectures de régression ont été évaluées sur le jeu de test après entraînement :

| Modèle | RMSE (Test) | $R^2$ Score (Test) | Analyse |
| :--- | :--- | :--- | :--- |
| **Support Vector Regressor (SVR Linéaire)** | 960,90 | 0,2750 | Incapable de capturer les non-linéarités complexes de la demande. |
| **Régression Linéaire** | 863,17 | 0,4150 | Modèle de référence sous-ajusté (*underfitting*). |
| **Arbre de Décision (Decision Tree)** | 437,17 | 0,8499 | Bonne capture des seuils mais variance résiduelle élevée. |
| **Forêt Aléatoire (Random Forest)** | **317,28** | **0,9210** | **Meilleure performance globale par bagging d'arbres.** |

### Validation croisée (10-Folds)
Pour confirmer la robustesse du Random Forest, une validation croisée à 10 plis a été effectuée :
* **Score $R^2$ moyen** : $0{,}9215$
* **RMSE moyen** : $278{,}82$ vélos

---

## 4. Optimisation des Hyperparamètres (*Fine-Tuning*)

L'optimisation a été menée via `GridSearchCV` (1 620 combinaisons testées sur 5 folds) et confirmée via `RandomizedSearchCV` :

* **Paramètres optimaux retenus** :
  * `n_estimators` : 200
  * `max_depth` : 20
  * `min_samples_split` : 5
  * `min_samples_leaf` : 2
  * `max_features` : None

### Performance finale du modèle retenu

| Jeu de données | RMSE | $R^2$ Score |
| :--- | :--- | :--- |
| **Train set** | 134,89 | 0,9842 |
| **Test set** | **309,44** | **0,9248** |

L'écart maîtrisé entre train et test confirme l'absence de surapprentissage critique (*overfitting*) tout en garantissant un coefficient de détermination supérieur à 92 % sur des périodes temporelles jamais observées.

---

## 5. Exemple d'Inférence Opérationnelle

Test de prédiction sur une nouvelle observation simulée :
* **Scénario** : Samedi de juillet à 18h00, température de 20°C, humidité de 66 %, vent à 7 km/h, temps clair.
* **Volume prédit** : **2 414 vélos** (GridSearchCV) / **2 431 vélos** (RandomizedSearchCV).

---

## 6. Structure du Dépôt

```text
london-bike-demand-forecasting/
├── README.md                 # Synthèse méthodologique et résultats
├── london_merged.csv         # Données historiques de partage de vélos à Londres
└── Regression_case.ipynb     # Notebook complet (code, transformations et évaluations)
