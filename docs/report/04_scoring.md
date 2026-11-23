# 5. Scoring sémantique et recommandation (P4)

Le moteur transforme les réponses du questionnaire en **profil sémantique**, puis note chaque livre avec une formule explicite et **entièrement traçable**.

**Profil utilisateur.** Toutes les réponses deviennent du texte encodé par SBERT (`all-MiniLM-L6-v2`, vecteurs normalisés). La requête globale $q_{global}$ est la moyenne des embeddings des trois réponses libres. La moyenne, préférée à la concaténation, donne le même poids à chaque question et évite la troncature à 256 tokens. Si l'auteur préféré est trouvé dans le catalogue, on y mélange le centroïde de ses livres (0,7 / 0,3). Pour chaque bloc $b$ (Genre, Thèmes, Style, Atmosphère), on construit l'ensemble $P_b(u)$ des phrases de préférence : les phrases descriptives des labels cochés, le texte libre lié, et surtout les **Likert convertis en phrases**. Par exemple, une complexité de 5 devient « *a complex, dense, literary and demanding narrative* ». Une réponse numérique est ainsi comparée aux livres par son sens. Le niveau neutre (3) ne produit aucune phrase.

**Formule.** Avec $e_j$ l'embedding du livre $j$ et $L_b(j)$ ses labels attribués par P2 :

$$S_{desc}(j) = \cos(q_{global}, e_j) \qquad S_b(j) = \frac{1}{|P_b|}\sum_{p \in P_b(u)} \max_{l \in L_b(j) \cup \{e_j\}} \cos(p, l)$$

$$\tilde s = \operatorname{clip}\!\left(\frac{s - p_5}{p_{95} - p_5}, 0, 1\right) \qquad \text{Coverage}(j) = \frac{\sum_b W_b\, \tilde S_b(j)}{\sum_b W_b}$$

$$\text{Final}(j) = \frac{\alpha\, \tilde S_{desc}(j) + \beta\, \text{Coverage}(j) + \lambda\, \text{Pop}(j)}{\alpha + \beta + \lambda} - \mu \max\big(0, \cos(q_{avoid}, e_j) - \tau\big)$$

avec $\alpha = 0{,}5$, $\beta = 0{,}4$, $\lambda \in [0 ; 0{,}2]$, $\mu = 0{,}5$ et $\tau = 0{,}3$ (constantes dans `src/config.py`).

**Justification.** $\tilde S_{desc}$ compare deux textes libres (la demande et le résumé) : c'est le signal sémantique le plus riche, d'où le poids le plus fort. La **couverture**, exigée par le sujet, vérifie que le livre répond à *chaque* dimension demandée. Ses poids $W_b$ sont les Likert d'importance : c'est l'utilisateur qui décide. La règle « max puis moyenne » reprend l'exemple de la prof : chaque préférence est satisfaite par le meilleur label du livre, et le livre doit répondre à toutes. Ajouter le résumé $e_j$ au max rend le score robuste quand un label manque. La popularité reste un petit bonus réglé par un Likert. La division par $\alpha + \beta + \lambda$ garantit $\text{Final} \in [0,1]$ : les seuils *Excellent* (≥ 0,75), *Good* (≥ 0,55) et *Partial* gardent le même sens pour tous. Le texte « à éviter » est encodé **sans ses négations**, car SBERT gère mal la négation (« *no romance* » ressemble à « *romance* »).

**Calibration.** Les cosinus SBERT sont compressés, d'où la calibration par percentiles. Nous calculons $p_5$ et $p_{95}$ **sur les 300 candidats retrouvés pour 8 personas de référence**, et non sur tout le catalogue. En effet, le retrieval garde le top 2,5 % : tous les candidats dépassent le p95 du catalogue, et $\tilde S_{desc}$ vaudrait 1 pour tous. Elle est ensuite figée, donc comparable entre utilisateurs.

**Pipeline et explications.** `recommend(answers)` applique les filtres durs (public, format, auteur exclu), relâchés progressivement si aucun livre ne passe. Viennent ensuite un retrieval des 300 plus proches, un re-ranking vectorisé (numpy) et une diversification (un livre par auteur). Chaque recommandation contient son score, son libellé, la décomposition du score (la somme des contributions donne le score final), les labels communs, la phrase du résumé la plus proche de la demande et une **explication générée par template**.

**Résultats et limites.** Sur 12 000 livres avec MiniLM, la précision de genre@3 atteint 0,92 sur nos 8 personas, contre 0,75 avec la similarité globale seule ; augmenter le poids des thèmes fait bien monter le score Thèmes du top 10 (0,37 → 0,43). Les tests révèlent aussi des limites. Une **proximité de sujet n'est pas une pertinence** : pour le persona *dark academia*, le mot « secrets » fait remonter *My Secret: A PostSecret Book*, un recueil non-fictionnel que P2 a de plus étiqueté à tort *Thriller & Suspense*. Filtrer la non-fiction quand seuls des genres de fiction sont cochés corrigerait ce cas. Avec $\lambda \le 0{,}2$ et $\tau = 0{,}3$, le bonus popularité et la pénalité « à éviter » ne modifient pas le top 10 : leurs valeurs seront ajustées par l'ablation de P6. Enfin, la calibration repose sur 8 personas, et l'absence de données de série empêche le filtre « standalone ».
