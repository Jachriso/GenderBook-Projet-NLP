# Slides: Person 3, semantic engine (≈ 2 min 30 s)

Speaking order: after P2 (labels and enrichment), before P4 (scoring). Message to get across: **the system understands meaning, not just words, and we chose the model with measurements.**

---

## Slide 1: From text to meaning (≈ 45 s)

**Title:** Turning text into meaning

**Visual:** a simple pipeline
`blurb / user answer` → `minimal cleaning` → `SBERT (all-MiniLM-L6-v2)` → `vector of 384 numbers` → `cosine similarity`

**Bullets (short):**
- Every text (12,000 blurbs, user answers, label descriptions) becomes a 384-dimension vector
- Close meaning = close vectors = high cosine
- Minimal cleaning for SBERT: *"He is NOT happy"* keeps its negation
- 12,000 books encoded once (7 min), then a search takes a few milliseconds

**Speaker notes:** Introduce the idea that a vector captures meaning. Mention that the cache is shared by the whole team (P4 for scoring, P5 for the map). Do not go into implementation details.

---

## Slide 2: Semantic vs keywords, the core result (≈ 1 min)

**Title:** Finding a book without a single shared word

**Visual:** bar chart `docs/figures/p3_encoder_mrr.png` (MRR by method) + one example in large font:
> Query: *"a parentless lad learns spells from an old wizard…"*
> Expected: *Magician* (blurb: *"orphan boy apprenticed to a master magician"*)
> TF-IDF 305th → GloVe 20th → **SBERT 1st**

**Small table:**

| | TF-IDF | GloVe | SBERT MiniLM | SBERT MPNet |
|---|---|---|---|---|
| Found in top 10 (/7) | 0 | 2–4 | 6 | 6 |
| MRR | 0.00 | 0.22–0.25 | 0.54 | 0.79 |

**Speaker notes:**
- TF-IDF needs shared words, so it finds nothing: this is why the brief asks for semantic analysis, not keyword matching.
- GloVe knows synonyms but averages words: *Lysistrata* 352nd with GloVe, 1st with SBERT.
- Be honest: on a rare proper noun (*Habsburg*), GloVe wins.
- MPNet is better but 8 times slower: MiniLM stays the default, the switch is one parameter.
- Caveat in one sentence: 7 illustrative queries, the full evaluation is P6's.

---

## Slide 3: What the vector space looks like, and its limits (≈ 45 s)

**Title:** A map of 12,000 books

**Visual:** `docs/figures/p3_umap_genres.png` (UMAP map coloured by genre, star = "boy wizard" query)

**Bullets:**
- Genres form regions: Romance, Fantasy, crime, non-fiction
- 79 % of a book's 10 nearest neighbours share a genre vs 38 % at random
- Raw cosines are compressed: the 300 candidates are all above the catalogue's 95th percentile → justifies P4's calibration
- Limits: commercial blurbs (*Pride and Prejudice* blurb = praise, no plot), negations, 256-token truncation (19.5 % of blurbs, no gain from chunking)

**Speaker notes:** Point to the star landing near the Fantasy region. Transition to P4: "the map shows genres overlap, so P4 uses the genre labels explicitly in the score".

---

## Backup slide (questions only)

**Visual:** `docs/figures/p3_cosine_distribution.png`
- Why cosine? Angle, not length; equivalent to Euclidean distance on normalised vectors (same ranking), one matrix product for 12,000 books
- Why not chunking? No gain on our queries, +26 % encoding time
- Why MiniLM and not MPNet? 8× slower on CPU, consistent with P2's enrichment and P4's calibration

## Demo idea (if P5's live demo allows it)

Type *"a boy discovers he is a wizard and goes to a school of magic"* and show the similarity map with the user point, plus the "closest sentence" under a recommendation.
