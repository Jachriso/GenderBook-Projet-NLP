# 10. Generative AI bonus and conclusion

## 10.1 Generative AI bonus

<!-- TWO VERSIONS: keep the one matching the final state of the project. -->

<!-- Version A: bonus not implemented -->
The assignment optionally suggests adding a generative AI component (LLM-written explanations, chatbot, RAG or agent). We chose to **devote our time to the mandatory requirements** and their evaluation. The system is designed to accommodate it easily: each recommendation already contains, in structured form, everything an LLM should receive according to the assignment (user profile, scores, contributing criteria, closest sentence of the blurb). An `llm_explain.py` module could rephrase the template-based explanation from these elements only, automatically falling back to the template if no API key is available.

<!-- Version B: bonus implemented (to adapt)
We added an LLM-written explanation (`src/llm_explain.py`, [Gemini free tier / Ollama]).
The LLM only receives the profile, the scores, the recommended books and the contributing
criteria, with the instruction to invent nothing and to stay faithful to the scores. It can be
switched on with a toggle in the application; without an API key or in case of error, the
application shows the template-based explanation. The recommendation engine therefore works
entirely without an LLM, as the assignment requires. [Add an example: template vs LLM explanation.]
-->

## 10.2 Conclusion

We built a book recommendation system that **understands preferences expressed in natural language**. A reader describes what they would like to read, a book they loved and the atmosphere they are looking for. The system compares the meaning of these answers with 12,000 blurbs and returns three books, each with a score, the criteria that explain it and the sentence of its blurb closest to the request.

Three choices shaped the project:
1. **Turn everything into text**: categories, labels and even numerical answers are expressed as sentences, then compared through embeddings. This is what makes it possible to match "a boy who discovers he is a wizard" with *Fantasy* without any word in common.
2. **A decomposable score**: global similarity, coverage of each block, and bonuses or penalties are combined transparently, with weights partly chosen by the user. Each recommendation can thus be explained by a template, with no black box.
3. **Evaluation at every step**: label quality on unseen books, encoder comparison on queries with no word in common, ablation of the formula, and testing of the complete system on personas.

On 10 test personas built independently of the tuning, the complete system with MiniLM recommends books of the requested type (genre-precision@3 of 0.97), covers the requested themes (0.97) and always respects the user's explicit choices (excluded author, format). The coverage score is what keeps the right type of book: with the blurb similarity alone, the genre-precision drops to 0.80. Compared with the lexical baseline, the main contribution of semantic embeddings is **robustness**: when the request is rephrased, MiniLM keeps most of its top 3 (Jaccard 0.63) whereas TF-IDF + LSA changes most of it (0.28), and MiniLM never returns a book the user has already read. MPNet brought no clear gain on the complete system for a 4 times slower recommendation, so MiniLM was kept for the application.
<!-- Add one sentence with the judged precision once data/eval/judgments.csv is filled in. -->

The results confirm the value of semantic analysis over keyword matching: on queries that share no word with the blurb of the expected book, TF-IDF finds none of the books, whereas SBERT places most of them in its top ten (section 4.7). They also show the limits of the approach: the quality of commercial blurbs, the difficulty of negations, and the fact that **topical proximity is not always relevance**.

Beyond the technical result, the project taught us that a reliable NLP system depends as much on the data and on testing concrete cases as on the choice of model. The most important errors were found by trying the system as a user would, then fixed and documented.
