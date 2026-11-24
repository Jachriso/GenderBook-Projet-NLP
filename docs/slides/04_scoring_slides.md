# P4 slides: semantic scoring and recommendation

## Slide 1: From the questionnaire to the semantic profile
- Key idea: **everything becomes a sentence and then an embedding**, even the Likerts
  - complexity 5 → "a complex, dense, literary and demanding narrative"
  - pace 1 → "a slow, contemplative and quiet story"
- 4 sets of preferences P_b (Genre, Themes, Style, Atmosphere) + global query q_global (average of the free-text answers + favourite author)
- "To avoid" text encoded without the negations (SBERT handles "no" poorly)
- Visual: diagram answers → sentences → vectors, and the profile radar (block_affinities)

## Slide 2: The scoring formula
- Final = [0.5·S̃_desc + 0.4·Coverage + λ·Pop] / (0.9 + λ) − 0.5·max(0, cos(avoid) − 0.3)
- Coverage = Σ W_b·S̃_b / Σ W_b (assignment formula, weights = importance Likerts)
- S_b = "max then mean": each preference finds its best label in the book
- p5/p95 calibration on the candidates: makes the cosines readable without saturating the top
- Visual: the formula + stacked bars of a book's contributions

## Slide 3: Example computed by hand + explanation
- One persona (dark academia) → 1 book of the top: S_desc, S_Genre, S_Themes, S_Style, S_Atmosphere, coverage, final score (table of section 6 of the notebook)
- The generated explanation: closest sentence of the blurb, shared themes, strongest and weakest criterion
- Consistency: 3 contrasting personas → 3 very different tops (children's fantasy / Nordic noir / philosophical essay)
- Mini-ablation: S_desc alone vs Coverage alone vs full formula (genre precision@3)
- Oral message: "The score is not a black box: every point comes from a similarity we can show."
