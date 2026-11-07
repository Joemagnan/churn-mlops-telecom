# Projet fil rouge MLOps — Prédiction du churn télécom

## Objectif
Créer une chaîne ML reproductible : données → entraînement → comparaison → sauvegarde du pipeline → API FastAPI → tests → CI GitLab → Docker → monitoring minimal.

## Jeu de données choisi
**IBM Telco Customer Churn** (7 043 clients, 21 colonnes, cible `Churn`, 26,5 % de churn).
Source officielle : https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv
Ce jeu est un **exemple pédagogique d'opérateur fictif**, et non un fichier commercial réel.
Empreinte SHA-256 du fichier utilisé : `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91` (recopiée dans `models/metadata.json` à chaque entraînement).

## Installation (Python 3.12)
```bash
python -m venv .venv
# Windows : .venv\Scripts\activate
# Linux/macOS : source .venv/bin/activate
pip install -r requirements-dev.txt
```
`requirements.txt` contient uniquement ce dont l'API a besoin (c'est ce que l'image Docker installe) ; `requirements-dev.txt` y ajoute pytest, httpx et matplotlib. Les versions sont figées : le modèle joblib doit être rechargé avec les mêmes bibliothèques que celles qui l'ont produit.

## Entraînement sur le VRAI jeu IBM (connexion Internet)
```bash
python -m src.download_data
python -m src.train
python -m src.analyse      # courbes ROC/PR, étude du seuil, variables les plus influentes
```
Résultats obtenus le 07/11/2026 (régression logistique retenue, jeu test de 1 409 clients) :

| ROC-AUC | PR-AUC | Précision | Rappel | F1 | Accuracy |
|---|---|---|---|---|---|
| 0,841 | 0,632 | 0,503 | 0,783 | 0,613 | 0,737 |

Le détail est dans `reports/metrics.json` et `reports/analyse.json`, la figure dans `reports/courbes_roc_pr.png`.

## Démonstration hors ligne (données SYNTHÉTIQUES)
```bash
python -m src.demo_data
python -m src.train --demo
```
La version `1.0.0-demo` ne doit PAS être utilisée pour conclure à une performance métier.
**Mise à jour :** cette commande enregistre désormais le modèle synthétique dans `models/demo/` et ses métriques dans `reports/demo/`. Elle ne modifie plus le modèle IBM. Ne construisez une image destinée à un usage réel qu’avec des artefacts réels validés.

## Tests
```bash
python -m pytest -q
```
18 tests : schéma des données, absence de fuite de la cible, valeurs manquantes, API (validation des entrées, prédiction, 503 sans modèle, journal sans donnée personnelle, journal inaccessible), monitoring, et 3 tests de l'extension dérive (lot simulé, sélection des colonnes, colonnes manquantes).

## Démarrer l'API
```bash
uvicorn src.api:app --reload
```
Visiter `http://127.0.0.1:8000/docs` ; tester `/health`, `/model-info`, `/predict`.
```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d @sample_request.json
```
Sous PowerShell, utiliser `curl.exe` à la place de `curl` si nécessaire.

## Docker
Après entraînement et en présence de `models/churn_pipeline.joblib` :
```bash
docker build -t churn-api:1.0.0 .
docker run -d --name churn-api -p 8000:8000 -e ENABLE_INFERENCE_LOG=1 -v churn-monitoring:/app/monitoring churn-api:1.0.0
docker ps                                   # la colonne STATUS doit afficher (healthy)
docker exec churn-api python -m src.monitor
```
Le volume `churn-monitoring` conserve le journal d'inférences quand le conteneur est supprimé puis recréé. Sans journalisation, `docker run --rm -p 8000:8000 churn-api:1.0.0` suffit.
Les artefacts sont générés localement et inclus lors du build ; ne jamais accepter de modèle joblib provenant d'une source non fiable.

## CI GitLab
Le fichier `.gitlab-ci.yml` réalise :
1. `unit_tests` : les tests automatiques (`python:3.12-slim`, comme l'image de l'API) ;
2. `smoke_training` : un entraînement de démonstration sur données synthétiques ;
3. `drift_smoke` : le rapport Evidently sur une dérive simulée ;
4. `docker_smoke` : sur la branche principale et à la demande, la construction de l'image puis l'attente de l'état `healthy` dans Docker-in-Docker (runner privilégié nécessaire ; job facultatif).

Les quatre jobs de cette version ont été rejoués en local le 07/11/2026, chacun dans son image Docker et à partir d'un `git clone` du dépôt : tous passent. Ce rejeu ne remplace pas une exécution sur un vrai runner GitLab.
**Le CI ne prouve pas qu'un modèle IBM a été entraîné ou déployé en production.** L'image construite dans le job Docker manuel contient le modèle de démonstration. Pour aller plus loin, prévoir un accès aux données approuvé, un registre de modèles, la publication dans un registre de conteneurs, des secrets CI et un environnement cible.

## Monitoring
Activer `ENABLE_INFERENCE_LOG=1` pour produire `monitoring/inferences.jsonl` puis :
```bash
python -m src.monitor
```
Seuls les scores, le niveau de risque, la version du modèle et l'horodatage sont journalisés. Si le journal ne peut pas être écrit, l'API répond quand même et signale le problème dans ses logs. Les vrais contrôles en production devraient intégrer la dérive des données, le taux d'erreur, la latence, la qualité des prédictions après observation du churn réel, des alertes et un processus de réentraînement.

## Limitations et éthique
- Validation/test stratifiés, sans fuite de cible et prétraitements ajustés sur l'entraînement.
- Une probabilité n'est pas une certitude ; aucune résiliation n'est déduite automatiquement.
- Les variables démographiques du jeu IBM exigent des tests d'équité et une justification métier avant usage réel.
- Le seuil par défaut 0,5 est pédagogique ; il doit être optimisé selon les coûts de rétention (voir `reports/analyse.json`).
- Les 11 valeurs vides de `TotalCharges` concernent des clients d'ancienneté nulle : elles sont imputées par la médiane, alors que 0 serait plus logique (effet négligeable ici).
- Les données IBM sont transversales, donc pas adaptées à une vraie validation chronologique sans données datées.
- Le monitoring est simplifié (pas de dérive statistique complète, ni Prometheus).


## Extension MLOps — MLflow, Evidently, GitLab CI/CD

**Cette extension conserve le modèle IBM de base et sépare désormais les expériences synthétiques.**
Le fichier `data/telco_churn.csv` n'est pas redistribué : l'obtenir via `python -m src.download_data`.
Les résultats historiques sont ceux déclarés dans le projet original et n'ont pas été recalculés ici sur le CSV absent.

### 1. Tests et démonstration (sans MLflow/Evidently)
```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m src.demo_data
python -m src.train --demo
```
Le modèle synthétique va désormais dans `models/demo/`, les métriques dans `reports/demo/` et le jeu de référence dans `monitoring/demo/`. Il **ne remplace plus** le modèle IBM dans `models/`.

### 2. Expériences MLflow
```bash
python -m pip install -r requirements-mlops.txt
python -m src.train --demo --mlflow
mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000
```
Consulter `http://127.0.0.1:5000`. Pour l'expérience IBM : `python -m src.download_data` puis `python -m src.train --mlflow`. Les données doivent être correctement obtenues avant entraînement.
MLflow et Evidently sont figés (3.18.0 et 0.7.23). MLflow 3.18 enregistre les modèles scikit-learn au format skops, qui refuse par défaut certains types internes du pipeline : `src/mlflow_tracking.py` déclare explicitement les deux types concernés.

### 3. Monitoring de dérive Evidently
```bash
python -m src.drift_demo --reference monitoring/demo/reference.csv
python -m src.drift_report --reference monitoring/demo/reference.csv --current monitoring/demo/current_SIMULE.csv --output reports/data_drift_demo.html
```
Ouvrir `reports/data_drift_demo.html`. **La dérive est provoquée artificiellement pour démontrer le mécanisme**, pas mesurée sur une population réelle. Pour surveiller des clients réels, remplacer le fichier courant par un export autorisé, sans ID ni cible, et intégrer ensuite des labels différés pour mesurer la qualité des prédictions. Vérifier la conformité et la durée de conservation des données.

### 4. CI GitLab réelle
Créer un dépôt privé GitLab et y pousser ce dossier. Dans **Build > Pipelines**, observer `unit_tests`, `smoke_training`, `drift_smoke`. Télécharger `reports/data_drift_demo.html` parmi les artefacts. Le job `docker_smoke` est manuel, réservé aux runners configurés pour Docker-in-Docker privilégié. Il **n'effectue aucun déploiement de production** et ne publie pas d'image avec un modèle synthétique.

### 5. Distinction essentielle
- **Test local** : vérifie Python, FastAPI et la génération de références.
- **MLflow réel** : exige l'installation de MLflow et une exécution avec `--mlflow`.
- **Evidently réel** : exige son installation ; le rapport HTML est produit par ses calculs.
- **Pipeline GitLab réel** : exige un push sur GitLab et un runner disponible ; aucun statut vert n'est affirmé sans cette étape.
- **Production** : nécessite une gestion des secrets, un registre privé, des artefacts IBM approuvés et signés, un suivi des labels réellement observés et une revue de sécurité.
