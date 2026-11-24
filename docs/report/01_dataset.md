# 2. Dataset: origin, structure and preparation

*Section written by person 1 (Data Engineering). Code: `src/data_loader.py`, `src/marketing.py`, `scripts/build_dataset.py`, `notebooks/01_data_preparation.ipynb`.*

## 2.1 Origin

The reference catalogue comes from the public **Book Depository Dataset** published on Kaggle (`sp1thas/book-depository-dataset`). It was obtained by scraping the website of the British online bookseller Book Depository.
It contains **1,109,383 rows**, i.e. a 1.1 GB `dataset.csv` file, and four lookup tables (`authors.csv`, `categories.csv`, `formats.csv`, `places.csv`).
Each row is a book **edition** described by a title, authors and categories (stored as lists of identifiers), an **editorial description** in natural language, the language, the format, the recommended age, the publication date and the **Goodreads ratings** (average and number of ratings).
The download is automated with `kagglehub`, without an API key. The raw data is not versioned.

## 2.2 Why this dataset suits semantic recommendation

The assignment requires a **contextual** analysis of the reader's preferences, not simple keyword matching. A rich text describing the content of each book is therefore needed. The `description` column is filled in for 93% of the books, with a median of 441 characters. The **categories** (≈ 2,900 labels from the booksellers' nomenclature) provide thematic blocks, and the **Goodreads ratings** a quality and popularity signal. On the other hand, the dataset contains no themes, narrative style or keywords, unlike the structure suggested by the assignment (`BookID | Genre | Title | Author | Synopsis | Themes | Style | Keywords`). These attributes are **derived by NLP** in the next step (section 3).

## 2.3 Findings of the exploratory analysis

- **Languages**: 89% of the books are in English.
- **Formats**: about 105,000 audio CDs, as well as calendars, maps and games.
- **Generic descriptions**: 12,485 texts are repeated at least 3 times and cover 141,070 books. They are mostly automatic public-domain reprints ("*This book is a facsimile reprint…*", "*Print on Demand*") and notebooks. These texts describe no content and would produce identical embeddings.
- **Marketing**: about 7% of the descriptions start with press quotes (*"A chilling thriller" --Booklist*) or banners (*#1 NEW YORK TIMES BESTSELLER*), and many end with a *Praise for…* section. These passages describe a book's success, not its content, and would artificially bring best-sellers closer together in the vector space.
- **Multiple editions**: the same work appears in dozens of editions, up to 101 for *Romeo and Juliet*. Titles vary (*Philosopher's Stone* / *Sorcerer's Stone*, *Dune : Deluxe Edition*, *Cat's Eye* / *Cats Eye*).
- **Authors**: the same author has a different identifier for each spelling (*J.K. Rowling* / *J. K. Rowling* / *J K Rowling*), and some editions contain typos (*Jane Austin*).
- **Target audience**: `for-ages` is missing for 93% of the books.

## 2.4 Preparation pipeline

| Step | Books left | Justification |
|---|---:|---|
| Raw dataset | 1,109,383 | |
| English language | 986,575 | English SBERT model, English interface |
| Title and description present | 952,244 | nothing to encode otherwise |
| Paperback / Hardback format | 832,881 | removes audio CDs, calendars, toys |
| ≥ 50 Goodreads ratings | 232,731 | reliable ratings, rich metadata; numeric filter applied before the costly text cleaning |
| Raw description ≥ 300 characters | 136,531 | a text that is too short carries too little meaning |
| Not only generic text | 135,877 | generic sentences removed in 1,018 texts; those that had nothing else are discarded |
| ≥ 300 characters without marketing | 135,257 | enough content remains once marketing is removed |
| "Leisure reading" categories | 110,905 | fiction + narrative non-fiction |
| At least one named author | 110,883 | needed for display and deduplication |
| Without box sets / bundles | 110,632 | a box set is not *one* reading experience |
| Without spin-off products | 110,303 | pop-up books, film "making of" books, official guides |
| Merging editions | 95,877 | one work = one row |
| Without shared descriptions | 93,724 | remaining generic publisher texts |
| ≤ 8 books per author | 80,718 | catalogue diversity |
| Top candidates by popularity | 13,799 | 15% margin before the language check |
| Description really in English (langdetect) | 13,709 | `lang` describes the book, not always its description |
| **Final catalogue** | **12,000** | the most rated works |

In the EDA, most of the rows discarded at the "description" stage are removed by the **length threshold** (≈ 295,000 raw descriptions under 300 characters), not by the generic-text detection (≈ 42,000). The two filters are logged separately.

**Text cleaning.** The cleaning is deliberately **minimal**: decoding HTML entities, removing tags and URLs, Unicode (NFKC) and whitespace normalisation. We do not lowercase, remove stopwords or lemmatise, because contextual models use the full sentence (negations, punctuation, function words). This classic preprocessing is reserved for the TF-IDF and GloVe baselines. The text with minimal cleaning only is kept in `description_raw` for traceability.

**Generic text, sentence by sentence.** The generic patterns ("facsimile reprint", "public domain", "print on demand"…) are applied **sentence by sentence**: the sentence concerned is removed and the rest is kept. The sentence "*The original book has been in the public domain since 1956*" disappears, but the blurb of *The Wizard of Oz* around it is kept. A book whose description was only generic text falls below the 300-character threshold and is discarded.

**Marketing removal (`src/marketing.py`).** The difficulty is knowing **where the attribution** following a press quote **ends**. In *"A premium delivery." --People Meghan doesn't know Agatha…*, "People" is a magazine but "Meghan" is already the plot. A "sequence of capitalised words" rule cannot tell the difference. The module therefore relies on knowledge taken from the dataset itself:
- a list of **press publications** (People, Booklist, Kirkus…);
- the **631,000 author names** of `authors.csv`, to recognise "--Stephen King";
- the **1.1 million titles** of the dataset, to know where "*author of The Statistical Probability of Love at First Sight*" ends.

When nothing is recognised, a cautious fallback only removes a likely name. Two safety rules complete the approach:
- a banner is only removed if it forms a **standalone segment**, never if it is followed by a lowercase word ("*The #1 New York Times bestselling debut novel that…*" is kept);
- a quote without an identifiable source after the dash is a quote **from the story** and is left untouched.

*Evaluation*: a description that starts with a lowercase letter signals a cut sentence. The first, purely regex-based version produced 235 of them. The current version produces 28, against 9 in the raw text. The remaining cases are mostly **leftover attributions** that were not removed ("*via Twitter*"), a deliberately cautious error.

**Categories.** Each label is classified by transparent regex rules as *fiction*, *narrative non-fiction* (biographies, history, true crime, popular science, philosophy…), *excluded* (cookery, computing, textbooks, law, medicine, dictionaries…) or *neutral*. A book is kept if it has at least one "leisure" category and at least as many leisure categories as excluded ones. The domain (*Fiction* / *Non-fiction*) is the majority class.

**Canonical author spelling.** Each author is replaced by the most frequent spelling among the variants with the same normalised key (`jkrowling`). The search utilities also compare these keys. Without this, 119 authors of the catalogue had several spellings, and excluding J.K. Rowling only removed one Harry Potter out of 7.

**Spin-off products.** A **precise** list of patterns (pop-up book, *Page to Screen*, official or unofficial guides, film art books, sticker books, fill-in journals), complemented by "corporate" authors (Warner Bros, Scholastic…), removes 329 products. Broad patterns ("guide to", "the art of") would have removed *The Art of War* or *The Hitchhiker's Guide to the Galaxy*.

**Merging editions.** A **union-find** algorithm links two rows if:
- (a) they have the same normalised title (without edition mentions, apostrophes deleted, hyphens replaced by spaces, compared **without spaces**) and at least one shared author;
- (b) or their Goodreads ratings are nearly identical (Goodreads aggregates ratings per work: number of ratings within ±2%, average within ±0.05) **and** their titles are similar: Jaccard of the meaningful words ≥ 0.5, or one title contained in the other if the shorter one has at least 3 words (or 2 words and a shared author).

In each group, the most rated standard edition is kept. The main author is the **majority author across all the editions**, which fixes isolated typos (*Pride and Prejudice* is correctly credited to Jane Austen). For example, the 25 English editions of *Harry Potter* book 1 are grouped into a single work.

The rules were refined by inspecting the groups whose titles have no word in common:
- a first version accepted "same author" without a similar title and merged different Lee Child novels (1,606 doubtful groups);
- the inclusion rule then merged *Civil War* (Lucan) with *Spider-Man: Miles Morales: Civil War II*.

Today 14 groups of this kind remain out of ≈ 70,000, and they are real variants (*Blindside* / *Blind Side*, translations). No title + author duplicate remains in the catalogue.

**Target audience.** We use `for-ages` when it exists (minimum age < 12 → *Children*, 12-17 → *Young Adult*). Otherwise, the audience is inferred from the categories (*Teenage*, *Young Adult* → *Young Adult*; *Children's*, *for Kids*, *Picture Books* → *Children*), and *Adult* by default.

**Final selection.** We keep the 12,000 most rated works, with at most 8 books per main author. The main author is the author of the list who is most frequent in the corpus, so as not to count illustrators and translators. The assignment recommends at least 60 books. 12,000 books ensure good coverage of all profiles, an embedding computation of a few minutes on CPU, rich editorial descriptions and recognisable books, which makes qualitative evaluation and the demo easier.

## 2.5 Resulting catalogue

- **12,000 works**, 5,860 distinct main authors, 1,074 Book Depository categories.
- **Domain**: 9,336 *Fiction* (78%), 2,664 *Non-fiction* (22%).
- **Audience**: 9,339 *Adult*, 1,440 *Children*, 1,221 *Young Adult*.
- **Popularity**: between 5,185 and 7.4 million Goodreads ratings (median: 14,143).
- **Descriptions**: 143 words on average. About 80% of the `embed_text` texts (title + description, cleanly truncated at ≈ 1,500 characters) fit entirely within the 256 tokens of `all-MiniLM-L6-v2`, and all of them within the 384 tokens of `all-mpnet-base-v2`, measured with the model's tokenizer.

**Columns produced** (`data/processed/books_clean.parquet`): `book_id`, `title` (without edition mentions), `authors` (canonical spelling), `authors_str`, `main_author`, `categories`, `description` (NLP text), `description_raw`, `pub_year`, `rating_avg`, `rating_count`, `bestsellers_rank`, `format`, `for_ages`, `audience`, `domain`, `is_fiction`, `image_url`, `url`, `isbn13`, `n_editions`, `popularity`, `n_words`, `embed_text`.

**Popularity score**, used in the recommendation formula:

$$\text{Pop}(j) = \frac{\text{rating\_avg}_j}{5}\cdot\frac{\log(1+\text{rating\_count}_j)}{\log(1+\max_k \text{rating\_count}_k)} \in [0,1]$$

The logarithm prevents a handful of best-sellers with several million ratings from crushing the rest of the catalogue.

**Quality checked by 52 unit tests** (`tests/test_data_loader.py`). They cover in particular the cases flagged in review: author variants, press quotes, punctuation duplicates, spin-off products.

## 2.6 Limitations

- **Popularity bias and English-language bias**: the catalogue reflects the sales of a British online bookseller. This bias is mitigated by the per-author cap and by the questionnaire's "best-sellers ↔ hidden gems" setting.
- **Residual marketing**: a little marketing built into sentences remains ("*#1 bestselling classic*"), deliberately kept when removing it would cut the sentence.
- **Imperfect rules**: category classification, the inferred audience and spin-off detection are rule-based and make a few errors at the margin.
- **`pub_year`** is the publication year of the edition, not the period of the story. The period and atmosphere are therefore determined semantically from the description (*Atmosphere* block).
