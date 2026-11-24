# Appendix: compliance with the assignment requirements

*Each requirement of the assignment is linked to the code that implements it and to the section of the report that documents it.*

| Requirement | Implementation | Section | Status |
|---|---|---|---|
| **Req. 1**: web questionnaire | `app/streamlit_app.py`, `app/questions.py` | 6 | ☑ |
| Free text (≥ 1 mandatory) | 4 open questions, one of which requires at least 30 characters | 6 | ☑ |
| Likert | complexity, pace, tone, popularity, importance of the 4 blocks | 6 | ☑ |
| Guided questions | favourite author, characters or plot, exclusion of the favourite author | 6 | ☑ |
| Multiple choice / checkboxes | genres, themes, period, audience, formats | 6 | ☑ |
| Responses stored in a structured format | timestamped JSON in `data/responses/` | 6 | ☑ |
| **Req. 2**: reference dataset | `data/processed/books_enriched.parquet` (12,000 books) | 2, 3 | ☑ |
| Categories / thematic blocks | `data/reference/blocks.json`: Genre, Themes, Style, Atmosphere | 3 | ☑ |
| Natural-language descriptions | cleaned blurbs (`description`, `embed_text`) and a descriptive sentence per label | 2, 3 | ☑ |
| Keywords | 5 keywords per book (KeyBERT principle) | 3 | ☑ |
| Relationships categories ↔ books | genres, themes, styles, atmospheres and `label_scores` per book | 3 | ☑ |
| Origin, structure, preprocessing documented | `notebooks/01_data_preparation.ipynb`, `filtering_report.csv` | 2 | ☑ |
| **Req. 3**: semantic engine | `src/preprocessing.py`, `src/embeddings.py` | 4 | ☑ |
| Preprocessing and normalisation | minimal cleaning (SBERT) and classic cleaning (baselines) | 2, 4 | ☑ |
| Word embeddings (GloVe) | `src/encoders.py` (GloVe, plus TF-IDF and TF-IDF + LSA baselines) | 4 | ☑ |
| Contextual embeddings (SBERT) | `all-MiniLM-L6-v2`, compared with `all-mpnet-base-v2` | 4 | ☑ |
| Cosine similarity | `EmbeddingEngine.search`, vectorised scoring | 4, 5 | ☑ |
| Clustering (optional) | `src/semantic_map.py`: UMAP map and genre neighbourhood analysis | 4 | ☑ |
| Justification of the model and the measure | TF-IDF / GloVe / MiniLM / MPNet comparison, on retrieval (section 4) and on the complete system (section 7) | 4, 7 | ☑ |
| **Req. 4**: defined and justified score | `src/scoring.py`, `src/profile.py` | 5 | ☑ |
| Block aggregation and weights | "max then mean", weights $W_b$ from the Likert answers | 5 | ☑ |
| Thresholds | *Excellent* ≥ 0.75, *Good* ≥ 0.55, *Partial* | 5 | ☑ |
| Structured constraints | hard filters (audience, format, excluded author, fiction) | 5 | ☑ |
| **Coverage score** $\sum W_i S_i / \sum W_i$ | `Coverage(j)` in `src/scoring.py` | 5 | ☑ |
| **Req. 5**: ranked top 3 | `recommend(answers, top_k=3)` in `src/recommender.py` | 5 | ☑ |
| Semantic score per book | `final_score`, `match_label` | 5 | ☑ |
| Main criteria | shared labels and per-component contributions | 5 | ☑ |
| Template-based explanation | `src/explain.py` | 5 | ☑ |
| **Req. 6**: visualisation | `src/viz.py` | 6 | ☑ |
| User profile / summary | `UserProfile.summary`, profile radar | 5, 6 | ☑ |
| Block scores | "you vs the books" radar, contribution bars | 6 | ☑ |
| Similarity map | UMAP projection of the 12,000 books | 4, 6 | ☑ |
| Show that results come from semantic analysis | highlighted closest sentence, "How was this score computed?" panel, heatmap | 6 | ☑ |
| **Works without generative AI** | no LLM call in `recommend()` | 5, 10 | ☑ |
| **Deliverables**: commented notebooks | `notebooks/01` to `06` | all | ☑ |
| PDF documentation | this report | — | ☐ |
| Presentation and demo | slides + demo, all 6 members present | — | ☐ |
