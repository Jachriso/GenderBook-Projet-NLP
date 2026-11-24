# 4. Semantic engine: representations, similarity and model choice

*Section written by Person 3 (semantic engine). Code: `src/preprocessing.py`, `src/embeddings.py`, `src/encoders.py`, `src/semantic_map.py`, `scripts/p3_*.py`, `notebooks/03_semantic_engine.ipynb`.*

## 4.1 Role in the system

The semantic engine turns any text (a book blurb, a reader's free-text answer, the descriptive sentence of a label) into a **vector**, and measures how close two texts are in meaning with **cosine similarity**. Every other module relies on it: the enrichment (section 3) to assign themes and styles, the scoring (section 5) for the global similarity and the block scores, the application (section 6) for the similarity map, and the evaluation (section 7) to compare encoders.

The interface was agreed by the team (`EmbeddingEngine`):

| Method | Role |
|---|---|
| `encode(texts)` | minimal cleaning, SBERT encoding, normalised vectors (norm 1) |
| `book_matrix()` | (12,000 × 384) matrix of the books, encoded **once** and then reloaded from a disk cache |
| `label_matrix(block)` | vectors of the `blocks.json` labels (Genre, Themes, Style, Atmosphere) |
| `search(q, k)` | the k books closest to a query |
| `best_sentence(q, text)` | the sentence of the blurb closest to the query (explainability) |

## 4.2 Preprocessing: two cleanings depending on the model

The models have different needs, hence two separate functions.

- **For SBERT** (`preprocess_for_transformer`), the cleaning is **minimal**: HTML entities, leftover tags, URLs, Unicode normalisation, typographic quotes, repeated punctuation. Case, punctuation and function words are kept, because a contextual model reads the whole sentence: *"a story with **no** romance"* and *"a story with romance"* must stay different.
- **For TF-IDF and GloVe** (`preprocess_for_classic`), the classic pipeline is applied: lowercasing, spaCy tokenisation and lemmatisation (`en_core_web_sm`), removal of stopwords, punctuation and numbers.

The example *"He is NOT happy at first"* shows the difference: the bag-of-words version becomes `happy`, and the negation disappears. This is why the text sent to SBERT is barely preprocessed.

A third function, `split_sentences`, splits a blurb into sentences with a rule-based splitter (it does not break after "Dr."). It is used by `best_sentence`, which shows in the application **the passage of the blurb that justifies the recommendation**.

## 4.3 Three text representations

| Representation | Principle | Synonyms | Context |
|---|---|---|---|
| **TF-IDF** (lexical) | sparse vector of the words, weighted by their frequency in the text × their rarity in the catalogue | no | no |
| **GloVe** (static word embeddings, `glove-wiki-gigaword-100`) | average of the word vectors learned on Wikipedia; an IDF-weighted variant is also tested | yes | no |
| **SBERT** (contextual embeddings, `all-MiniLM-L6-v2`) | a transformer reads the whole sentence; it is fine-tuned so that two sentences with the same meaning get close vectors | yes | yes |

GloVe gives **a single vector per word**, whatever the context, and averaging loses word order and negations. SBERT produces one vector per sentence that takes each word in its context into account.

The four encoders share the same interface (`get_encoder("tfidf" | "glove" | "glove-tfidf" | "minilm" | "mpnet")`), so they can be compared with the same code, here and in the evaluation of section 7.

## 4.4 Similarity measure: cosine

$$\cos(a, b) = \frac{a \cdot b}{\lVert a\rVert\,\lVert b\rVert}$$

- Cosine measures the **angle** between two vectors, regardless of their length: a long blurb and a short blurb on the same topic stay close.
- It is the measure SBERT is trained with.
- Since all vectors are **normalised**, $\cos(a,b) = a\cdot b$ and the Euclidean distance satisfies $\lVert a-b\rVert^2 = 2 - 2\cos(a,b)$. **Both measures give exactly the same ranking** (checked numerically in the notebook). Cosine has the advantage of being computed for all 12,000 books with **a single matrix-vector product**.

At this scale, an exhaustive search takes only a few milliseconds: an approximate index (FAISS) would bring nothing.

## 4.5 Engine implementation

- **Aligned cache.** Encoding the catalogue takes about **7 minutes on CPU** (35 ms per book). The matrix is saved in `data/embeddings/` together with the list of `book_id` in the same order. On loading, if the identifiers no longer match (regenerated catalogue), the engine re-encodes instead of mixing up books.
- **Consistency with the enrichment.** `label_matrix` reuses the `encode_blocks` function of section 3: the label vectors are **identical** to those used to enrich the catalogue (including the v2 themes, averages of 4 sentences).
- **Lazy loading.** The model is only loaded at the first encoding, which makes importing the module instantaneous for the application.
- **Tests.** 42 unit tests (`tests/test_preprocessing.py`, `test_embeddings.py`, `test_encoders.py`, `test_semantic_map.py`) use a deterministic fake model: they check the logic (normalisation, cache, alignment, search, chunking) without downloading anything.

## 4.6 Long texts: truncation or chunking?

MiniLM only reads **256 tokens**. On the catalogue, the median is 178 tokens, the 95th percentile 308 and the maximum 419: **19.5 % of the blurbs exceed the limit**, and their end is ignored. Two strategies were compared on 7 paraphrased "famous book" queries, measuring the rank of the expected book among the 12,000:

| | Truncation | Chunking (average of chunks) |
|---|---|---|
| Expected book in the top 10 | **3 / 6** | 2 / 6 |
| MRR | **0.15** | 0.12 |
| Encoding time | **7 min** | 8 min 49 s (+26 %) |

**Decision: truncation.** No gain was observed, at a higher cost. Publishers put the hook at the beginning of the blurb, the end mostly contains praise and edition details, and averaging chunks dilutes the meaning. One exception is noted: *Pride and Prejudice* moves from 570th to 111th with chunking, as the end of its blurb describes more of the story.

## 4.7 Semantic vs keyword matching: a quantitative comparison

**Protocol.** 7 queries each describe one specific book of the sample (500 books) **without any word in common** with its blurb after lemmatisation (checked automatically). Example: *"a parentless lad learns spells from an old wizard while creatures from another dimension attack his homeland"* for *Magician* (R. E. Feist), whose blurb mentions an *"orphan boy apprenticed to a master magician"*. We measure the rank of the expected book (1 = found first, ≈ 250 = random) and the **MRR** (*Mean Reciprocal Rank*, average of 1/rank).

| | TF-IDF | GloVe | GloVe + IDF | **MiniLM** | MPNet |
|---|---:|---:|---:|---:|---:|
| Median rank | 305.5 | 20 | 9 | 3 | **1** |
| Found first | 0 / 7 | 1 / 7 | 1 / 7 | 3 / 7 | **5 / 7** |
| In the top 10 | 0 / 7 | 2 / 7 | 4 / 7 | **6 / 7** | **6 / 7** |
| MRR | 0.00 | 0.22 | 0.25 | 0.54 | **0.79** |
| Time (500 books, CPU) | 4 s | 65 s | 63 s | 26 s | ≈ 190 s |

- **TF-IDF finds nothing**: with no shared word, the similarity is zero. This shows that keyword matching is not enough.
- **GloVe understands synonyms**, but averaging common words loses the meaning of the sentence: *Lysistrata* is 352nd with GloVe and **1st with SBERT**.
- **SBERT is not better everywhere**: on a rare proper noun (*"Habsburg"*), GloVe ranks *The Good Soldier Svejk* 1st, MiniLM 4th.
- GloVe's time comes from spaCy lemmatisation, not from GloVe itself.

**MiniLM or MPNet?** MPNet (768 dimensions) is clearly better (MRR 0.79 vs 0.54) and fixes some of MiniLM's failures (*Dracula* moves from 32nd to 2nd). Its mean rank is nevertheless worse (8.4 vs 6.7) because of a single extreme case, hence the use of the median and the MRR. However, it is **about 8 times slower** on CPU (≈ 80 min for the catalogue). **MiniLM remains the default model** (responsive application, consistency with the enrichment and the calibration). Switching to MPNet requires no code change (`EmbeddingEngine("mpnet")`) and is a matter for the evaluation of section 7. The results are **reproducible**: two runs give the same ranks.

**Caveat.** 7 queries written by us **illustrate** the differences between the methods; they are not a statistical evaluation.

## 4.8 Scaling up: 12,000 books

On the full catalogue, with queries describing well-known books without naming them, the exact book is rarely first: *1984* is 2nd, *The Hunger Games* 5th, *The Hobbit* 8th, *Frankenstein* 27th, *Harry Potter* 32nd. The books ranked above them are nevertheless often **relevant** (*The Witch Boy*, *Magyk*, *Magician* for "a boy discovers he is a wizard"), which is what a recommender aims for.

Analysing the failures shows that they mostly come **from the data**:
- the blurb of *Pride and Prejudice* (570th) is **publisher praise** that does not tell the plot;
- the blurb of *Harry Potter and the Prisoner of Azkaban* describes **the illustrated edition**;
- the blurb of *Harry Potter and the Sorcerer's Stone* opens with **a series of negations** (*"He knows no spells, has never helped to hatch a dragon…"*), which embeddings handle poorly.

Consequence for section 5: a book ranked beyond 300th by the global similarity never enters the *retrieval* candidates, even with a perfect profile. The block scores, which depend less on the blurb alone, partly make up for this limitation.

## 4.9 Cosine distribution and calibration

Raw SBERT cosines are **compressed**. Over 14 queries × 12,000 books:

| | p5 | median | p95 |
|---|---:|---:|---:|
| Whole catalogue | −0.02 | 0.11 | 0.27 |
| 300 *retrieval* candidates | 0.27 | 0.33 | 0.41 |

**100 % of the candidates are above the catalogue p95** (query by query). A calibration $\tilde s = \mathrm{clip}\big((s-p_5)/(p_{95}-p_5), 0, 1\big)$ computed on the whole catalogue would therefore map almost all candidates to 1: the global similarity term would no longer separate anything. This measurement **justifies the choice made in section 5**, which calibrates on the candidates of reference personas. The engine therefore does not produce a competing calibration file (team decision).

## 4.10 Semantic map and genres

The similarity map of the application is a **UMAP** projection of the 12,000 vectors in 2D (cosine distance, 15 neighbours, `min_dist` = 0.1, fixed seed and frozen algorithm for reproducibility), saved in `data/embeddings/umap_2d.npy`. The user's point is placed at the similarity-weighted average position of their 10 closest books, which avoids saving the UMAP model.

On the map, genres occupy **regions**: Romance on one side, Fantasy on the other, Mystery and Thriller mixed together, Horror between Fantasy and crime, non-fiction grouped together (History, Philosophy & Religion, Biography). Literary Fiction is spread out: it is more a style than a subject.

Yet the **silhouette score** by main genre is close to 0 (−0.007). It requires clear-cut boundaries, whereas genres **overlap**: a book has up to three genres and some "genres" are audiences (Young Adult). A better-suited measure confirms what the map shows: **79 % of a book's 10 nearest neighbours share at least one of its genres, against 38 % for randomly drawn books (×2.1)**. Fiction genres are the most compact (Romance 88 %, Fantasy 84 %), non-fiction the most scattered (Biography 64 %, Cooking & Lifestyle 60 %): for non-fiction, **the subject matters more than the publishing genre**. This overlap justifies the explicit use of genre labels in the block scores.

## 4.11 Limitations

- **Blurb quality**: commercial texts, sometimes without a plot; some books cannot be found from their story.
- **Negations and surface words**: *"fight"* pulls up a fighter's autobiography for *The Hunger Games*, *"dead body"* pulls up forensic thrillers for *Frankenstein*.
- **Truncation** at 256 tokens for 19.5 % of the blurbs, with no measured gain from chunking.
- **Illustrative evaluation**: the comparisons in this section rely on 14 queries; the large-scale evaluation is the subject of section 7.
- **English only**: the multilingual model `paraphrase-multilingual-MiniLM-L12-v2` would allow answers in French, at the cost of a new evaluation.
