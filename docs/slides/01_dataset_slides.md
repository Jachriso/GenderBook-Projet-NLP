# Slides de la personne 1 : Dataset (≈ 2 min 30)

## Slide 1 : D'un million d'éditions à 12 000 œuvres
- **Visuel** : `docs/figures/01_filtering_funnel.png` (entonnoir de filtrage).
- **Message** : « Book Depository, c'est 1,1 million de lignes. Mais une analyse sémantique a besoin de **texte qui décrit le contenu** du livre. »
- Les trois pièges trouvés dans l'EDA, chacun en une ligne :
  1. **141 000 descriptions génériques** (« facsimile reprint », « print on demand ») : même texte, donc même embedding.
  2. **Citations de presse** en tête de 7 % des descriptions : elles parlent du succès du livre, pas de son contenu.
  3. **Éditions multiples** : *Harry Potter* tome 1 = 25 éditions, *Romeo and Juliet* = 101.
- **Chiffre clé** : 12 000 œuvres, 5 860 auteurs, 78 % fiction / 22 % non-fiction, 22 % jeunesse et YA.

## Slide 2 : Les choix qui comptent pour la recommandation
- **Visuel** : à gauche, un exemple AVANT / APRÈS de `strip_marketing` ; à droite, `docs/figures/01_description_tokens.png`.
- **Fusion des éditions** (union-find) : même titre + même auteur normalisé, OU mêmes notes Goodreads + titre proche. Raconter l'itération : la 1re version fusionnait deux romans de Lee Child, d'où la règle plus stricte (1 606 → 14 groupes douteux sur ≈ 70 000).
- **Retirer le marketing sans couper l'histoire** : dans « --People Meghan doesn't know Agatha », comment savoir que « Meghan » est l'intrigue ? On s'appuie sur ce que le dataset sait déjà : 631 000 noms d'auteurs, 1,1 M de titres, une liste de journaux. Résultat : 235 → 28 phrases coupées.
- **Une seule orthographe par auteur** (J.K. / J. K. / J K Rowling) : sinon « exclure Rowling » n'excluait qu'un Harry Potter sur 7.
- **Preprocessing minimal** volontaire : SBERT a besoin des stopwords et des négations (« *not* a happy ending »).
- **Public visé déduit** (for-ages manque à 93 %) : il sert de filtre dans le questionnaire.
- **Longueur en tokens** : 80 % des descriptions tiennent dans les 256 tokens de MiniLM, ce qui justifie la troncature à ≈ 1 500 caractères.
- **Transition vers P2** : « Il manque les thèmes, le style et l'ambiance : on va les **dériver par NLP**. »

## Questions probables du jury
- *Pourquoi pas tout le dataset ?* Le coût des embeddings, le bruit des livres peu connus et des descriptions pauvres. 12 000 ≫ 60 (minimum du sujet).
- *Le biais de popularité ?* Il est assumé et documenté, atténué par le plafond par auteur et par le Likert « pépites » du questionnaire.
- *Comment savez-vous que la fusion des éditions marche ?* 52 tests unitaires (Harry Potter fusionné, deux romans de Lee Child non fusionnés, Cat's Eye / Cats Eye…) et inspection des groupes dont les titres n'ont aucun mot commun (14 sur ≈ 70 000, tous de vraies variantes).
- *Comment avez-vous évalué le nettoyage du marketing ?* Indicateur simple : une description qui commence par une minuscule = phrase coupée. 235 avec la 1re version, 28 maintenant (9 dans le brut).
