# Appendix B: libraries and tools

*The assignment asks for the libraries used and the reasons for choosing them. All of them are free and open source, and the system runs on a laptop CPU without any paid API. The exact versions are in `requirements.txt`.*

| Library | Role in the project | Why this choice |
|---|---|---|
| **sentence-transformers** (with PyTorch) | contextual sentence embeddings (SBERT) for the blurbs, the label sentences and the user's answers | SBERT is trained so that two sentences with the same meaning get close vectors, which is exactly the comparison we need. `all-MiniLM-L6-v2` (384 dimensions) encodes the 12,000 books in about 7 minutes on CPU and a request in a few milliseconds; `all-mpnet-base-v2` (768 dimensions) is used for comparison (sections 4 and 7). |
| **scikit-learn** | TF-IDF baseline, truncated SVD (TF-IDF + LSA baseline), silhouette score | reference implementation of the classic lexical methods we compare SBERT with (section 4.7). |
| **gensim** | GloVe word embeddings (`glove-wiki-gigaword-100`) | the assignment requires static word embeddings; GloVe is the standard choice and gensim downloads the pre-trained vectors directly. |
| **spaCy** (`en_core_web_sm`) | tokenisation, lemmatisation and stopwords for the baselines, part-of-speech tags for the keywords, rule-based sentence splitting | fast and robust English pipeline; the noun-phrase filter removes verb + noun pairs from the keywords, and the sentence splitter handles abbreviations ("Dr.", "U.S.") for the closest-sentence explanation. |
| **pandas**, **pyarrow**, **numpy** | data preparation, parquet storage, vectorised scoring | parquet keeps list columns and types and is ten times faster to read than CSV; with normalised vectors, scoring 12,000 books is a single matrix-vector product in numpy (about 1 ms), so no approximate index (FAISS) is needed. |
| **rapidfuzz** | fuzzy matching of the favourite or excluded author typed by the user | tolerates typos ("stephen kng" → Stephen King) after an exact match on a normalised key. |
| **langdetect** | check that the description really is in English | the `lang` column describes the book, not always its blurb; a fixed seed makes the result reproducible. |
| **kagglehub** | download of the Book Depository dataset | public dataset, no API key needed: the whole catalogue can be rebuilt with one command. |
| **umap-learn** | 2D projection of the 12,000 embeddings (semantic map) | preserves the neighbourhood of each book better than PCA, with the cosine metric used everywhere else; the seed and algorithm are fixed for reproducibility. |
| **Streamlit** | web questionnaire and results page | recommended by the assignment for quick prototypes; it keeps the whole project in Python and caches the model and the catalogue between requests. |
| **Plotly** | radars, stacked contribution bars, semantic map, heatmap | interactive charts in the application, exported as PNG for the report. |
| **pytest**, **Jupyter** | unit tests and documented notebooks | the tests use small fake models and run in a few seconds without downloads; the notebooks document every step with comments, as required by the assignment. |

**What we deliberately did not use.** No large language model takes part in the recommendation: the scores and the explanations are computed from the embeddings and from templates, so the system works offline and every number can be traced back to a similarity (section 10.1). We also did not use a vector database (FAISS, Chroma): at 12,000 books an exact search is already instantaneous.
