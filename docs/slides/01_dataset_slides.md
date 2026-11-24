# Person 1 slides: Dataset (≈ 2 min 30)

## Slide 1: From a million editions to 12,000 works
- **Visual**: `docs/figures/01_filtering_funnel.png` (filtering funnel).
- **Message**: "Book Depository is 1.1 million rows. But a semantic analysis needs **text that describes the content** of the book."
- The three traps found in the EDA, one line each:
  1. **141,000 generic descriptions** ("facsimile reprint", "print on demand"): same text, therefore same embedding.
  2. **Press quotes** at the start of 7% of the descriptions: they talk about the book's success, not its content.
  3. **Multiple editions**: *Harry Potter* book 1 = 25 editions, *Romeo and Juliet* = 101.
- **Key figure**: 12,000 works, 5,860 authors, 78% fiction / 22% non-fiction, 22% children's and YA.

## Slide 2: The choices that matter for the recommendation
- **Visual**: on the left, a BEFORE / AFTER example of `strip_marketing`; on the right, `docs/figures/01_description_tokens.png`.
- **Merging editions** (union-find): same normalised title + same author, OR same Goodreads ratings + similar title. Tell the iteration: the 1st version merged two Lee Child novels, hence the stricter rule (1,606 → 14 doubtful groups out of ≈ 70,000).
- **Removing marketing without cutting the story**: in "--People Meghan doesn't know Agatha", how do we know that "Meghan" is the plot? We rely on what the dataset already knows: 631,000 author names, 1.1 M titles, a list of newspapers. Result: 235 → 28 cut sentences.
- **A single spelling per author** (J.K. / J. K. / J K Rowling): otherwise "exclude Rowling" only excluded one Harry Potter out of 7.
- Deliberately **minimal preprocessing**: SBERT needs the stopwords and the negations ("*not* a happy ending").
- **Inferred target audience** (for-ages is missing for 93%): it is used as a filter in the questionnaire.
- **Length in tokens**: 80% of the descriptions fit within MiniLM's 256 tokens, which justifies the truncation at ≈ 1,500 characters.
- **Transition to P2**: "The themes, the style and the atmosphere are missing: we will **derive them by NLP**."

## Likely questions from the jury
- *Why not the whole dataset?* The cost of the embeddings, the noise of little-known books and poor descriptions. 12,000 ≫ 60 (the assignment's minimum).
- *The popularity bias?* It is accepted and documented, mitigated by the per-author cap and by the questionnaire's "hidden gems" Likert.
- *How do you know that merging editions works?* 52 unit tests (Harry Potter merged, two Lee Child novels not merged, Cat's Eye / Cats Eye…) and inspection of the groups whose titles have no word in common (14 out of ≈ 70,000, all real variants).
- *How did you evaluate the marketing cleaning?* A simple indicator: a description that starts with a lowercase letter = cut sentence. 235 with the 1st version, 28 now (9 in the raw text).
