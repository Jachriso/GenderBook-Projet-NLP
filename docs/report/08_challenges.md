# 8. Challenges and solutions

Each module was reviewed by another member before being integrated, and the full system was tested on personas. These reviews revealed several problems, most of which would not have been visible by looking at average scores alone. We group them into four families.

## 8.1 Very noisy real-world data

| Problem | How we found it | Solution |
|---|---|---|
| **The same work in dozens of copies**: *Harry Potter* appeared 25 times, *Romeo and Juliet* 101 times, with different titles (*Philosopher's* / *Sorcerer's Stone*, *Dune : Deluxe Edition*) | EDA | De-duplication with *union-find*: same normalised title and same author, or near-identical Goodreads ratings and similar titles (section 2) |
| **The same author under several spellings** (*J.K. Rowling*, *J. K. Rowling*, *J K Rowling*) | Code review: searching for "jk rowling" returned **one book out of seven**, and excluding this author removed only one Harry Potter book. 119 authors of the catalogue were affected. | One canonical spelling per author, chosen from a normalised key; unit test added |
| **Press quotes in the blurbs** (*"A chilling thriller" --Booklist*), which would pull best-sellers closer together in the vector space | EDA, then review: the first version cut off the start of the plot ("*--People **Meghan** doesn't know Agatha*" lost "Meghan") | The `marketing.py` module uses the dataset's 631,000 author names and 1.1 million titles to find where an attribution ends |
| **Generic reprint texts** ("*facsimile reprint*", "*public domain*") | EDA | Removal sentence by sentence rather than dropping the whole description, to keep the surrounding blurb |

**Lesson.** On a scraped dataset, the quality of the encoded text matters as much as the choice of model. An SBERT model faithfully encodes whatever it is given, press reviews included.

## 8.2 Attributes missing from the dataset

The assignment suggests describing each book by its themes, style and period. The dataset contains none of these. Annotating 12,000 books by hand was impossible. P2 therefore **derived them through semantic similarity** between each blurb and a descriptive sentence for each label (section 3). The first versions produced telling errors:

- **wrongly forced period**: forcing a period on every book classified *Harry Potter* as "Medieval". A period is now assigned only when the signal is clear;
- **wrongly inferred audience**: more than 600 adult books received the "Young Adult" genre through semantic proximity. These genres now come only from the `audience` column;
- **theme sentences that were too abstract**, far from the concrete vocabulary of blurbs: each theme is now represented by the average of four phrasings.

These fixes were only possible because P2 measured label quality on hand-annotated books, 40 of which were never used for tuning.

## 8.3 Pitfalls of semantic similarity

**SBERT cosines are compressed.** Over the whole catalogue, 90% of the cosines lie between −0.02 and 0.27. A naive calibration on the whole catalogue would have given the maximum score to every candidate, because the 300 retrieved books all exceed the catalogue's 95th percentile (P3's measurement, section 4.9). P4 therefore calibrates on the candidates of reference personas, and P3's analysis justifies this choice.

**Negation is poorly understood.** For SBERT, "*no romance*" looks like "*romance*". The "to avoid" text is therefore encoded **without its negations** and used as a penalty, instead of being added to the query.

**Topical proximity is not relevance.** For a reader asking for a psychological novel about students "*hiding dark secrets*", the system suggested *My Secret: A PostSecret Book*, a collection of anonymous confessions. The word "secret" brought the texts together, but it is not a novel. Solution: when the user ticks only fiction genres, non-fiction and collections are filtered out automatically.

**An explanation must never contradict the request.** A reader who asked for a slow pace was shown "*Fast-paced page-turner fits the kind of writing you asked for*". The label was indeed assigned to the book, but it had not been requested. The explanation now only cites requested labels (ticked, derived from the Likert answers, or among the strongest affinities) and excludes those that contradict an answer.

**A display detail can mislead the user.** A score of 0.7497, displayed as "0.75", was labelled *Good* although the *Excellent* threshold is 0.75. The label is now computed on the rounded score, as displayed.

## 8.4 Six people working in parallel

**Dependencies between modules.** Scoring (P4) depends on the embeddings (P3), which depend on the catalogue (P1). To avoid waiting, the team fixed the **interface contracts** from the start (catalogue columns, JSON formats, dataclasses). P1 first delivered a 500-book sample in the final format. P4 and P5 developed against a fallback engine and a stub `recommend()`, later replaced by the real modules.

**An invisible failure during the demo.** P4's fallback engine silently took over if P3's SBERT engine failed (model not downloaded, no network). During the demo, we would have shown degraded results without knowing it. P3's review flagged this, and the application now displays an explicit warning.

**Computation time.** Encoding the 12,000 blurbs takes about 7 minutes on CPU. The embeddings are cached on disk together with the list of identifiers: if the catalogue changes, the engine detects it and re-encodes instead of mixing up books.

**Comparing encoders on equal terms.** To compare TF-IDF, GloVe and SBERT inside the complete pipeline, every encoder has to plug into the same engine. Integration tests showed two obstacles. TF-IDF returns a sparse matrix that the engine cannot use (made dense, it would take about 2.4 GB for 12,000 books). And the engine cached its embeddings and calibration under a model name that MiniLM and MPNet shared, so the second model would have silently reused the first one's vectors. P3 added a dense TF-IDF + LSA variant (256 dimensions) and gave each encoder a unique name, with a test checking that every dense encoder plugs into the engine.

**Environment.** Some libraries (gensim) cannot yet be installed with the latest Python version. We recommend Python 3.11 or 3.12 in the README.

**Evaluating fairly.** The first run of the evaluation favoured the lexical baseline for a wrong reason: in each test persona, the "book I loved" answer described one of the expected books, so a method that simply matched the user's words found it, and was rewarded for recommending a book the user had **already read**. We fixed a rule from the `loved_book` texts only, without looking at the results: the book described there is never counted as relevant. We also added blind judgments of every recommended book, because hit@3 and MRR only know the few books we listed in advance. Having the same books judged by two judges from the team showed a second difficulty: the two judges agree only slightly (kappa 0.18), mostly because one is stricter than the other, so "relevant" itself has to be defined and its subjectivity reported (section 7.2).

**Review process.** Every contribution went through a *pull request* reviewed and tested by another member before merging. Almost all the problems described in this section were found this way, by testing concrete cases ("what does the system return for *jk rowling*?") rather than only reading the code.
