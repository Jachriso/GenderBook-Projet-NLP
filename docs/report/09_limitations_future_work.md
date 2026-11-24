# 9. Limitations and future work

## 9.1 Limitations

**Data.**
- **Catalogue bias**: the books come from a British online bookseller and were selected among the most rated on Goodreads. The system therefore mostly recommends well-known English-language books. The cap of 8 books per author and the "best-sellers ↔ hidden gems" setting mitigate this bias without removing it.
- **Commercial blurbs**: some blurbs describe the edition or quote reviews instead of telling the story. The book then becomes hard to find from its content (*Pride and Prejudice* ranks 570th for a query describing its plot, section 4.8).
- **Residual data errors**: building the evaluation personas revealed errors that the cleaning rules did not catch. Some works still exist in several editions (*Frankenstein* four times, *Middlemarch* three times). The translator or the author of the foreword is sometimes taken as the main author (*Crime and Punishment* attributed to Richard Pevear, *The Remains of the Day* to Salman Rushdie). A few adult books are classified for children (*The Alchemist*, *Good Omens*), and some genre labels are wrong (*The Hunger Games* labelled *Historical Fiction*). These errors affect the hard filters and the genre scores at the margin.
- **Missing information**: the dataset does not say whether a book belongs to a series, so the "standalone book" preference is collected but cannot be applied. `pub_year` is the year of the edition, not the period of the story.

**Model.**
- **Truncation**: MiniLM only reads 256 tokens, and 19.5% of the blurbs are longer. Chunking brought no measurable gain (section 4.6).
- **Negations and surface words**: despite the handling of the "to avoid" text, a strong word can override the meaning ("*fight*" pulls up a boxer's autobiography for a request close to *The Hunger Games*).
- **Topic versus genre and register**: a request for a literary psychological novel also receives psychological thrillers that share its topic (section 7.6), and a marketing sentence left in a blurb can still drive a recommendation (*The Letter* recommended to a feel-good reader because of "*Perfect for fans of The Notebook*").
- **Geographical atmosphere**: the Atmosphere block describes periods and moods, not places, so a "Nordic" request is poorly captured (section 7.6).
- **English only**: the questionnaire must be answered in English.

**Scoring and evaluation.**
- **Imperfect labels**: on 40 unseen books, themes reach an F1 of 0.40 (section 3). These themes were annotated by a single person, with no inter-annotator agreement measure.
- **Hand-set weights**: $\alpha$, $\beta$, $\lambda$, $\mu$ and $\tau$ are set by hand, not learned. The ablation (section 7.4) confirms the value of the coverage score, but 10 personas are too few to separate the effect of the popularity, the "to avoid" penalty and the calibration. The calibration itself relies on 8 personas.
- **Small-scale evaluation**: 10 test personas and a few dozen queries show trends, not statistically robust measurements. With only 2 to 5 hand-picked books per persona, hit@3 and MRR are very sparse (section 7.2). Above all, **no real reader has judged the recommendations**. Relevance was judged by one team member (4 personas) and by an LLM (all 10), and the two agree only slightly (Cohen's kappa 0.18 on 44 books, section 7.2): the judged precision depends on who judges, and part of the final labels comes from a model rather than a person.
- **Automatic fiction filter**: it only applies if the user ticks fiction genres exclusively. A user who mixes fiction and non-fiction may still receive a book that is close in topic but off the mark.

## 9.2 Future work

**Short term (without changing the architecture).**
- **Switch to MPNet**: it is clearly better than MiniLM on queries with no word in common (MRR 0.79 vs 0.54, section 4.7), and slightly better on the blind judgments of the complete system (judged precision@3 0.60 vs 0.50, section 7.2). It is about 8 times slower, but the catalogue is encoded only once. The change takes one line (`EmbeddingEngine("mpnet")`), provided the enrichment and the calibration are recomputed.
- **Add places to the Atmosphere block** (Scandinavia, Victorian England, university campus…) to capture geographical requests.
- **Re-rank the top 300 with a cross-encoder**, which reads the request and the blurb together and judges fine-grained relevance better, at the cost of a few hundred milliseconds.

**Medium term.**
- **Measure with real readers**: have students test the application, collect a rating for each recommendation ("relevant / not relevant") and tune the weights on this feedback rather than by hand. Several judges per book would also give a proper inter-annotator agreement.
- **Multilingual interface** with `paraphrase-multilingual-MiniLM-L12-v2`, to accept answers in French.
- **Enrich the data**: series and places from open sources (Open Library, Wikidata), more narrative blurbs.
- **Hybrid approach**: combine semantic similarity with collaborative filtering ("readers who liked this book also liked…") if usage data were available.
