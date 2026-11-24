# Slides P5 : application et visualisations

## Slide 1 : Un questionnaire qui capte le sens, pas des cases
- 17 questions en 5 étapes, **les 5 types exigés** : texte libre (Q1 obligatoire ≥ 30 car.), cases à cocher, choix multiple, Likert 1-5, questions guidées
- Les labels viennent de `blocks.json` (même référentiel que le moteur) ; la bulle ⓘ montre la phrase encodée
- Le Likert devient des **phrases** et des **poids W_b** → pas de scoring purement numérique
- Sortie : `answers.json` horodaté (contrat d'équipe), téléchargeable et rechargeable
- Visuel : 2 captures côte à côte (`05_screen_1_free_text.png`, `05_screen_4_likert.png`) + petit encadré JSON

## Slide 2 : Rendre chaque recommandation explicable
- Carte par livre : score + badge, **phrase du résumé la plus proche**, critères communs, explication
- 4 visualisations, une couleur par bloc partout :
  - radar profil (ce que le système a compris de moi)
  - radar vous vs livre (= score de couverture)
  - barres empilées (d'où vient chaque point du score)
  - carte UMAP du catalogue + heatmap mots ↔ labels (preuve « pas de keyword matching »)
- Visuel : `05_score_breakdown.png` + `05_similarity_map.png`
- Puis : **démo live** (voir `05_demo_script.md`)
