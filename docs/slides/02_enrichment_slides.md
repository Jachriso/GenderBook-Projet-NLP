# Slides P2 : référentiel et enrichissement sémantique

## Slide 1 : Un référentiel pour décrire les livres
- Problème : le dataset n'a ni thèmes, ni style, ni mots-clés ; catégories bruitées (1 078)
- Solution : taxonomie `blocks.json` : 4 blocs (Genre 21, Thèmes 20, Style 10, Atmosphère 15)
- Idée clé : chaque label = une **phrase descriptive** encodée en SBERT (« boy wizard school » ↔ Fantasy sans le mot)
- Visuel : schéma des 4 blocs avec 2 exemples chacun

## Slide 2 : Enrichissement par analyse sémantique
- Genres : règles + embeddings sur les catégories (table relue sur le top 100)
- Thèmes / Style / Atmosphère : zero-shot par embeddings, **z-score par label** + seuil calibré
- Mots-clés : principe KeyBERT (candidats TF-IDF → cosinus → MMR)
- Visuel : heatmap Genre × Thème (`docs/figures/02_heatmap_genre_theme.png`)
- Chiffres de validation (60 livres annotés) : precision@k, au moins un label correct
- Bonus oral : 1 bon exemple et 1 erreur commentée
