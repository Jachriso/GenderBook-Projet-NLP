# 1. Introduction and objectives

## 1.1 Problem

Choosing a book is a **semantic** decision: a reader is not looking for "a book containing the word *secret*", but for "a psychological novel about students hiding a crime, with a melancholic atmosphere". Yet booksellers' search engines mostly rely on commercial categories and keywords. They cannot match "an orphan boy who discovers he is a wizard" with a book filed under *Fantasy* if those words do not appear in its record.

This project addresses **theme 3 of Annex I** of the assignment, *Book Recommendation Based on Literary Preferences*. We built a web application that collects a reader's preferences in natural language, compares them with the content of 12,000 books through semantic analysis, and recommends the **three most suitable books**, each with a score and an explanation.

## 1.2 Objectives

The assignment sets six common requirements, which structure this report:

| Requirement | What we built | Section |
|---|---|---|
| Req. 1: varied web questionnaire | 17 Streamlit questions: free text, Likert, guided, multiple choice, checkboxes | 6 |
| Req. 2: structured reference dataset | catalogue of 12,000 books (Book Depository), enriched with genres, themes, styles, atmospheres and keywords | 2 and 3 |
| Req. 3: semantic analysis engine | preprocessing, TF-IDF (and TF-IDF + LSA), GloVe and SBERT, cosine similarity | 4 |
| Req. 4: justified scoring mechanism | global similarity, calibrated block scores, weighted coverage score, popularity, "to avoid" penalty | 5 |
| Req. 5: explained top 3 | score, main criteria, closest sentence of the blurb, template-based explanation | 5 and 6 |
| Req. 6: visualisation and interpretation | radar charts, contribution bars, UMAP semantic map, heatmap | 6 |

One principle drives the whole system: **every answer becomes text, then a vector**. Even a numerical answer is turned into a sentence before being compared with the books (a complexity of 5 becomes "*a complex, dense, literary and demanding narrative*"). The analysis therefore stays semantic from end to end, as the assignment requires, and never boils down to adding up ratings.

## 1.3 Choice of domain and dataset

Theme 3 is part of Annex I, so it required no further approval. Books lend themselves well to semantic analysis: every book has a **natural-language blurb**, and literary preferences are naturally expressed in sentences (themes, moods, style).

We chose the **Book Depository Dataset** (Kaggle, 1.1 million editions) for the richness of its descriptions and its Goodreads ratings. It is very noisy (audio CDs, automatic reprints, multiple editions, marketing) and contains neither themes nor style: section 2 describes the cleaning, and section 3 how these attributes are **derived with NLP**. The final catalogue of 12,000 books is far above the minimum of 60 books recommended by the assignment.

## 1.4 Architecture

```
Streamlit questionnaire ──► answers.json (saved in data/responses/)
                                 │
                                 ▼
                       build_profile()   free texts + Likert answers turned into sentences
                                 │        ──► SBERT embeddings per block
                                 ▼
 books_enriched.parquet ──► 1. hard filters (audience, format, excluded authors, fiction)
 catalogue embeddings        2. retrieval: the 300 books closest to the request
 blocks.json                 3. re-ranking: full formula (blurb + coverage
                                + popularity − "to avoid" penalty)
                             4. diversity: one book per author
                                 │
                                 ▼
                       Top 3: score, criteria, closest sentence, explanation
                                 │
                                 ▼
                       Visualisations: radars, contributions, UMAP map, heatmap
```

The code is organised into independent modules (`src/`), each with a commented notebook (`notebooks/01` to `06`) and unit tests. The application (`app/streamlit_app.py`) calls a single function, `recommend(answers, top_k=3)`.

**Stack.** Python, pandas, `sentence-transformers` (`all-MiniLM-L6-v2` by default, compared with `all-mpnet-base-v2`), scikit-learn (TF-IDF), gensim (GloVe), spaCy (baseline preprocessing), UMAP, Plotly and Streamlit.

## 1.5 Team organisation

The work was split into six roles of comparable workload. The **interfaces between modules were fixed from the start**: catalogue columns, format of `blocks.json` and `answers.json`, shared dataclasses. Everyone could therefore code in parallel against sample data. Each contribution was integrated through a *pull request* reviewed by another member before being merged.

| Member | Role | Responsibility | Section |
|---|---|---|---|
| DUTERTRE Maxime | P1: Data Engineer | dataset cleaning, catalogue, EDA | 2 |
| BENLEMQADEM Hajar | P2: Reference framework and enrichment | block taxonomy, genres, themes, keywords, validation | 3 |
| NOMO NGAH Marie | P3: Semantic engine | preprocessing, embeddings, encoder comparison | 4 |
| FAUTRAD Annabelle | P4: Scoring and recommendation | profile, formula, explanations | 5 |
| PARRILLA Jade | P5: Application and visualisations | questionnaire, results page, charts | 6 |
| EBOBISSE Suzy | P6: Evaluation and integration | evaluation, code reviews, integration, report | 1, 7 to 10 |


## 1.6 Outline

Section 2 presents the dataset and its preparation, section 3 the reference framework and the semantic enrichment, section 4 the embedding engine and the choice of model, section 5 the scoring formula and how recommendations are generated, and section 6 the application. Section 7 evaluates the system, section 8 discusses the challenges we faced, section 9 its limitations, and section 10 concludes. The appendix links each requirement of the assignment to its implementation.
