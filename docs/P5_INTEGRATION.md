# Partie P5 : état, intégration et dépendances

## Fichiers livrés par P5

| Fichier | Rôle |
|---|---|
| `app/streamlit_app.py` | application (questionnaire 5 étapes + page résultats en 5 onglets) |
| `app/questions.py` | contenu des 17 questions, personas de démo, validation, `answers.json` (sans Streamlit, testable) |
| `app/stub_recommender.py` | remplaçant **provisoire** de `recommend()` (même contrat), supprimable quand P4 a livré |
| `app/__init__.py` | permet `from app import ...` |
| `src/viz.py` | les 5 figures (radar profil, radar comparé, barres empilées, carte 2D, heatmap) + export PNG |
| `scripts/prepare_app.py` | pré-calcule embeddings + carte 2D et teste les personas (à lancer avant la démo) |
| `scripts/capture_screens.py` | captures d'écran automatiques pour le notebook 05 et les slides (optionnel, Playwright) |
| `notebooks/05_app_and_visualization.ipynb` | notebook commenté (à exécuter une fois P4 intégré) |
| `tests/test_viz.py`, `tests/test_app_questions.py` | 16 tests (`python -m pytest tests -q`) |
| `docs/report/05_app_visualization.md` | section 6 du rapport |
| `docs/slides/05_app_slides.md`, `docs/slides/05_demo_script.md` | plan des 2 slides + script de la démo |
| `.streamlit/config.toml` | thème clair de l'app |

Lancement : `python -m scripts.prepare_app` (une fois) puis `streamlit run app/streamlit_app.py` depuis la racine.
Lien direct vers un persona : `http://localhost:8501/?persona=Young%20fantasy%20fan`.

## Intégration avec P4 (branche `p4-scoring`) : testée, compatible

Testé le 24/11 en combinant `origin/p4-scoring` et les fichiers P5 : les 3 personas passent par le vrai
`recommend()`, l'app n'affiche plus la bannière « stub » et toute la suite de tests passe (157 tests).
Adaptations faites côté P5 pour suivre P4 :
- **Likert popularité** : sens de P4 (`lambda_from_likert`) : **1 = pépites, 5 = best-sellers** (libellés et personas mis à jour) ;
- `match_label` = `"Excellent"` / `"Good"` / `"Partial"` → l'app affiche « Excellent match » ;
- contribution `"Avoid penalty"` reconnue (alias dans `viz.normalize_key`) ;
- formule affichée = celle de P4 (normalisée par α + β + λ) ;
- l'app réutilise le moteur de `get_recommender()` : un seul modèle SBERT en mémoire, catalogue aligné ;
- heatmap : lignes = `profile.block_phrases` (3 phrases max par bloc).

Rien de bloquant à attendre de P4. Seul point : **merger `p4-scoring` dans `main`** ; tant que ce n'est pas fait,
la branche `p5/streamlit` (partie de `main`) tourne avec le stub. `guided.standalone_only` est lu par P4 mais le
dataset n'a pas d'info de série : la case reste, c'est une préférence douce.

## Ce que j'attends de P3

- Rien de bloquant : `EmbeddingEngine` fonctionne avec l'app.
- Si tu produis `data/embeddings/umap_2d.npy` (notebook 03), écris aussi `umap_2d_book_ids.json` (liste des `book_id` dans l'ordre des lignes) : sinon l'app vérifie seulement le nombre de lignes. Sans fichier, l'app calcule elle-même la carte (UMAP, ~1 min) et la met en cache.
- Rappel : les `.npy` sont dans `.gitignore`, donc chacun doit lancer `python -m scripts.prepare_app` (ou le notebook 03) sur sa machine.

## Ce que j'attends de P6

- `requirements.txt` : passer à `streamlit>=1.36` (j'utilise `vertical_alignment` dans `st.columns`). Optionnel : `playwright` pour les captures.
- README : ajouter la section « Lancer l'application » (commandes ci-dessus) et cocher P5 quand c'est mergé.
- Évaluation : les personas de `app/questions.py` (`DEMO_PERSONAS`, `build_answers`) donnent des `answers.json` valides ; le bouton « Results (JSON) » exporte profil + recommandations pour l'analyse.
- Rapport : ma section est `docs/report/05_app_visualization.md` (section 6 du plan) ; figures `docs/figures/05_*.png` (générées par le notebook 05).
- Vidéo de secours de la démo : je l'enregistre une fois P4 intégré (voir `05_demo_script.md`).

## Points remontés aux autres (non bloquants)

- P1 : quelques doublons d'édition restent visibles (*1984* apparaît 3 fois : « 1984 », « 1984 Nineteen-Eighty Four », « 1984 Deluxe Edition ») et certains noms d'auteurs gardent un titre (« Prof Bernhard Schlink »). La diversité « 1 livre par auteur » de P4 masque le premier point dans le top 3.
- Couvertures : les `image_url` pointent vers cloudfront ; si une image ne charge pas, l'app n'affiche simplement pas de couverture (pas d'erreur). À vérifier sur une vraie connexion (je n'ai pas pu tester le chargement des images).

## Après l'intégration de P4 (ma checklist)

- [ ] Après merge de P4 dans `main` puis de `main` dans `p5/streamlit` : vérifier « Recommender: P4 recommender » dans la barre latérale
- [ ] `python -m scripts.prepare_app` sans erreur, 3 personas cohérents
- [ ] Exécuter le notebook 05 (Run all) → figures `docs/figures/05_*.png`
- [ ] `python -m scripts.capture_screens` → captures `05_screen_*.png` pour le notebook et les slides
- [ ] Adapter les phrases du script de démo aux vrais résultats, puis enregistrer la vidéo de secours
- [ ] Supprimer `app/stub_recommender.py` (ou le garder comme repli : l'app l'utilise si `recommend` plante)
