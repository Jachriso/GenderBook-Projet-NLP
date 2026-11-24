# Script de la démo live (P5, ≈ 3 minutes)

> Les phrases entre guillemets sont des exemples : les adapter aux résultats réellement affichés une fois le moteur final (P4) branché.

## Avant la soutenance (la veille)
1. `python -m scripts.prepare_app` : encode les livres et calcule la carte (sinon le premier lancement prend plusieurs minutes).
2. `streamlit run app/streamlit_app.py`, ouvrir `http://localhost:8501`, faire un premier passage complet (le modèle est alors chargé en mémoire).
3. Vérifier dans la barre latérale : « Recommender: **P4 recommender** » (et non « stub »).
4. **Plan B** : enregistrer une vidéo de ce script (OBS ou l'enregistreur d'écran du système) et la garder sur le bureau.
5. Garder un onglet ouvert sur `http://localhost:8501/?persona=Dark%20academia%20reader` (formulaire pré-rempli).

## Déroulé

| Temps | Ce que je clique | Ce que je dis |
|---|---|---|
| 0:00 | Page d'accueil, étape 1 | « Voici le questionnaire. L'utilisateur décrit librement ce qu'il veut lire : c'est le cœur de l'analyse sémantique. » |
| 0:10 | Barre latérale → persona *Dark academia reader* → **Fill form** | « Pour gagner du temps, je charge un lecteur type. Il veut un roman psychologique sur des étudiants rongés par la culpabilité, dans une vieille université. » |
| 0:25 | **Next** → étape 2, survoler la bulle ⓘ d'un thème | « Cases à cocher : genres et thèmes. Chaque label correspond à une phrase, c'est elle que le modèle compare au résumé des livres. » |
| 0:40 | **Next** → étape 3, puis étape 4 | « Choix multiple : époque du récit, public, formats. Puis les échelles de Likert : elles deviennent des phrases et des poids, pas de simples nombres. » |
| 0:55 | **Next** → étape 5 | « Questions guidées : l'auteur favori, Donna Tartt, est retrouvé dans le catalogue ; on l'exclut pour découvrir quelqu'un de nouveau. » |
| 1:05 | **See my recommendations** | « Les réponses sont sauvegardées en JSON, puis le moteur compare la demande aux 12 000 livres. » |
| 1:15 | Onglet **Top 3**, livre n°1 | « Pour chaque livre : le score, le niveau de correspondance, et surtout la phrase du résumé la plus proche de la demande. Les puces colorées sont les critères communs. » |
| 1:40 | Ouvrir **How was this score computed?** | « La formule et chaque contribution : similarité au résumé, couverture des quatre blocs pondérée par l'utilisateur, popularité, pénalité. La somme donne le score. » |
| 2:00 | Onglet **Your profile** | « Ce que le système a compris : les thèmes Morality & guilt, l'atmosphère Academic & campus, alors que l'utilisateur n'a jamais écrit "campus". » |
| 2:20 | Onglet **Semantic map** (survoler l'étoile n°1) | « Chaque point est un livre, placé selon le sens de son résumé. Les genres forment des régions. L'utilisateur, la croix, tombe au milieu de ses recommandations. » |
| 2:40 | Onglet **Under the hood** | « La preuve que ce n'est pas du mot-clé : la case foncée relie "old college with libraries" et le label Academic & campus, sans mot commun. » |
| 2:55 | (si le temps le permet) persona *Young fantasy fan* → **Results ▶** | « Un profil totalement différent donne des recommandations totalement différentes. » |

## Si quelque chose casse
- Aucun résultat → bouton « Relax the filters » (et le dire : « les filtres durs sont respectés »).
- L'app est lente ou plante → passer à la vidéo sans s'excuser longuement.
- Bannière jaune « stub » visible → le moteur final n'est pas branché : vérifier que `src/recommender.py` est bien sur la branche.
