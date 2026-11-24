# P2 slides: taxonomy and semantic enrichment

## Slide 1: A taxonomy to describe the books
- Problem: the dataset has no themes, no style and no keywords; noisy categories (1,078)
- Solution: `blocks.json` taxonomy: 4 blocks (Genre 21, Themes 20, Style 10, Atmosphere 15)
- Key idea: each label = a **descriptive sentence** encoded with SBERT ("boy wizard school" ↔ Fantasy without the word)
- Visual: diagram of the 4 blocks with 2 examples each

## Slide 2: Enrichment through semantic analysis
- Genres: rules + embeddings on the categories (table reviewed on the top 100)
- Themes / Style / Atmosphere: zero-shot with embeddings, **per-label z-score** + calibrated threshold
- Keywords: KeyBERT principle (TF-IDF candidates → cosine → MMR)
- Visual: Genre × Theme heatmap (`docs/figures/02_heatmap_genre_theme.png`)
- Validation figures (60 annotated books): precision@k, at least one correct label
- Oral bonus: 1 good example and 1 commented error
