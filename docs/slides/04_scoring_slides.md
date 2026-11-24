# Slides P4 : scoring sémantique et recommandation

## Slide 1 : Du questionnaire au profil sémantique
- Idée clé : **tout devient phrase puis embedding**, même les Likert
  - complexité 5 → « a complex, dense, literary and demanding narrative »
  - rythme 1 → « a slow, contemplative and quiet story »
- 4 ensembles de préférences P_b (Genre, Thèmes, Style, Atmosphère) + requête globale q_global (moyenne des réponses libres + auteur préféré)
- Texte « à éviter » encodé sans les négations (SBERT gère mal « no »)
- Visuel : schéma réponses → phrases → vecteurs, et le radar du profil (block_affinities)

## Slide 2 : La formule de score
- Final = [0,5·S̃_desc + 0,4·Coverage + λ·Pop] / (0,9 + λ) − 0,5·max(0, cos(avoid) − 0,3)
- Coverage = Σ W_b·S̃_b / Σ W_b (formule du sujet, poids = Likert d'importance)
- S_b = « max puis moyenne » : chaque préférence trouve son meilleur label dans le livre
- Calibration p5/p95 sur les candidats : rend les cosinus lisibles sans saturer le top
- Visuel : la formule + barres empilées des contributions d'un livre

## Slide 3 : Exemple calculé à la main + explication
- Un persona (dark academia) → 1 livre du top : S_desc, S_Genre, S_Themes, S_Style, S_Atmosphere, couverture, score final (tableau de la section 6 du notebook)
- L'explication générée : phrase du résumé la plus proche, thèmes communs, critère le plus fort et le plus faible
- Cohérence : 3 personas contrastés → 3 tops très différents (fantasy jeunesse / polar nordique / essai philosophique)
- Mini-ablation : S_desc seul vs Coverage seule vs formule complète (précision de genre@3)
- Message oral : « Le score n'est pas une boîte noire : chaque point vient d'une similarité qu'on peut montrer. »
