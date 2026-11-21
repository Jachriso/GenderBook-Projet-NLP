"""
enrichment.py : enrichissement sémantique du catalogue (Personne 2).

Le dataset Book Depository ne contient NI thèmes, NI style, NI mots-clés, et ses
1 078 catégories sont bruitées. Ce module DÉRIVE ces informations par analyse
sémantique et produit ``books_enriched.parquet`` (colonnes du contrat PLAN §4.1) :

    genres        : list[str]  (top 3, bloc Genre)
    themes        : list[str]  (top 3, bloc Themes)
    styles        : list[str]  (top 2, bloc Style)
    atmospheres   : list[str]  (top 2 : 1 époque + 1 ambiance, bloc Atmosphere)
    keywords      : list[str]  (top 5, méthode type KeyBERT)
    label_scores  : str JSON   {bloc: {label: cosinus}}  (load_books() le reconvertit en dict)

Pipeline :
    1. blocks.json (taxonomie, 1 phrase descriptive par label)  -> load_blocks()
    2. catégories Book Depository -> genres (règles + embeddings) -> build_category_map()
    3. genres des livres                                          -> assign_genres()
    4. zero-shot par embeddings (Themes / Style / Atmosphere)     -> assign_labels()
    5. mots-clés (candidats TF-IDF + rerank embeddings + MMR)     -> extract_keywords()
    6. export                                                     -> save_enriched()

Tout encodeur exposant ``encode(list[str]) -> np.ndarray`` (L2-normalisé) convient :
``SbertEncoder`` ci-dessous, ou ``EmbeddingEngine`` de P3 quand il sera prêt.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Protocol

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Chemins et constantes
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]
BLOCKS_PATH = ROOT / "data" / "reference" / "blocks.json"
CATEGORY_MAP_PATH = ROOT / "data" / "reference" / "category_genre_map.csv"
EMB_DIR = ROOT / "data" / "embeddings"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

BLOCK_NAMES = ["Genre", "Themes", "Style", "Atmosphere"]

# Nombre de labels gardés par livre pour chaque bloc (contrat PLAN §4.1).
TOP_K = {"Genre": 3, "Themes": 3, "Style": 2, "Atmosphere": 2}

# Mots sans valeur sémantique pour décrire un livre (vocabulaire d'éditeur, mots grammaticaux espagnols/français...).
EXTRA_STOP = {"edition", "editions", "introduction", "includes", "include", "including", "novel", "novels",
              "book", "books", "series", "bestselling", "best", "selling", "author", "authors", "new", "york",
              "times", "reader", "readers", "read", "story", "stories", "life", "world", "first", "years",
              "year", "time", "isbn", "volume", "vol", "collection", "classic", "classics", "masterpiece",
              "translated", "translation", "penguin", "oxford", "press", "creates", "turns", "sets",
              "los", "las", "del", "della", "und", "der", "die", "das", "les", "des", "une", "est", "por",
              "para", "con", "una", "que", "el", "la", "le", "il", "di", "da"}

# Seuil z-score (par label, sur le catalogue) pour garder un label. Cf. threshold_report().
Z_MIN = 1.0

# Atmosphere = 1 époque + 1 ambiance : on distingue les deux familles de labels.
ERA_LABELS = {"Ancient world", "Medieval", "18th-19th century", "20th century",
              "Contemporary", "Future"}

# Pour le repli « genre le plus proche de la description », on respecte fiction / non-fiction.
NONFICTION_GENRES = {"Biography & Memoir", "History", "Science & Nature",
                     "Self-help & Psychology", "Philosophy & Religion",
                     "Business & Economics", "Cooking & Lifestyle"}
NEUTRAL_GENRES = {"Poetry & Drama", "Graphic Novels & Comics", "Humour", "Children's", "Classics"}

# Similarité minimale catégorie -> genre (embeddings) pour accepter l'association.
CATEGORY_MIN_SIM = 0.30

# Poids du vote « genre le plus proche de la description » face aux votes des catégories (1 vote chacune).
EMB_VOTE = 1.0


class Encoder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray: ...


# --------------------------------------------------------------------------- #
# 0. Encodeur minimal (en attendant EmbeddingEngine de P3)
# --------------------------------------------------------------------------- #
class SbertEncoder:
    """Encodeur SBERT minimal, chargement paresseux, sortie L2-normalisée.

    Compatible avec l'interface ``encode`` de ``EmbeddingEngine`` (P3).
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = "cpu"):
        self.model_name = model_name
        self.device = device
        self._model = None

    @property
    def model(self):
        if self._model is None:  # on ne charge le modèle qu'au premier appel
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def encode(self, texts: list[str], batch_size: int = 64, show_progress: bool = False) -> np.ndarray:
        emb = self.model.encode(
            list(texts), batch_size=batch_size, show_progress_bar=show_progress,
            convert_to_numpy=True, normalize_embeddings=True,
        )
        return emb.astype(np.float32)


def encode_cached(encoder: Encoder, texts: list[str], ids: list[str], cache_name: str,
                  batch_size: int = 64) -> np.ndarray:
    """Encode ``texts`` en utilisant un cache disque (.npy + .json d'ids pour l'alignement)."""
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    npy, ids_json = EMB_DIR / f"{cache_name}.npy", EMB_DIR / f"{cache_name}_ids.json"
    if npy.exists() and ids_json.exists() and json.loads(ids_json.read_text()) == list(ids):
        return np.load(npy)
    try:
        emb = encoder.encode(texts, batch_size=batch_size, show_progress=True)  # type: ignore[call-arg]
    except TypeError:  # encodeur sans ces arguments (ex. EmbeddingEngine)
        emb = encoder.encode(texts)
    np.save(npy, emb)
    ids_json.write_text(json.dumps(list(ids)))
    return emb


# --------------------------------------------------------------------------- #
# 1. Référentiel
# --------------------------------------------------------------------------- #
def load_blocks(path: Path | str = BLOCKS_PATH) -> dict[str, dict[str, str]]:
    """Charge et valide blocks.json : {bloc: {label: phrase descriptive}}."""
    blocks = json.loads(Path(path).read_text(encoding="utf-8"))
    missing = [b for b in BLOCK_NAMES if b not in blocks]
    if missing:
        raise ValueError(f"blocks.json : blocs manquants {missing}")
    for b, labels in blocks.items():
        for label, phrase in labels.items():
            if not isinstance(phrase, str) or len(phrase.split()) < 5:
                raise ValueError(f"blocks.json : phrase trop courte pour {b}/{label}")
    return blocks


def encode_blocks(encoder: Encoder, blocks: dict[str, dict[str, str]]):
    """Encode la PHRASE descriptive de chaque label (pas le mot seul).

    Retourne {bloc: (labels, matrice (L, d))}.
    """
    out = {}
    for b in BLOCK_NAMES:
        labels = list(blocks[b])
        out[b] = (labels, np.asarray(encoder.encode([blocks[b][l] for l in labels])))
    return out


# --------------------------------------------------------------------------- #
# 2. Catégories Book Depository -> Genres
# --------------------------------------------------------------------------- #
# (regex, genre, regex d'exclusion). Un nom de catégorie peut déclencher PLUSIEURS règles
# (ex. « Crime, Thriller & Mystery »). Ces règles couvrent les cas évidents ;
# le reste est traité par similarité d'embeddings.
CATEGORY_RULES: list[tuple[str, str, str | None]] = [
    (r"fantasy|wizard|sword (and|&) sorcery|epic fantasy", "Fantasy", None),
    (r"sci[ -]?fi\b|science fiction|dystopi|space opera|cyberpunk", "Science Fiction", None),
    (r"detective|mystery|whodunn|crime fiction|crime, thriller", "Mystery & Detective", None),
    (r"thriller|suspense|espionage|spy\b|legal thriller", "Thriller & Suspense", None),
    (r"horror|ghost|supernatural|gothic|vampire|zombie|occult", "Horror", None),
    (r"romance(?!\s+languages?)|romantic", "Romance", None),
    (r"historical fiction|historical novel|historical romance|sagas?\b", "Historical Fiction", None),
    (r"literary fiction|general fiction|contemporary fiction|modern fiction|literary", "Literary Fiction", None),
    (r"\bclassics?\b", "Classics", None),
    (r"young adult|teen", "Young Adult", None),
    (r"children|picture book|baby|toddler|early learning|story books|nursery|bedtime", "Children's", None),
    (r"humou?r|comedy|satire|joke|funny", "Humour", None),
    (r"biograph|autobiograph|memoir|diaries|true story", "Biography & Memoir", None),
    (r"\bhistory\b|military|ancient|medieval|\bwars?\b", "History", r"fiction|children|young adult"),
    (r"science|physics|biology|chemistry|nature|astronomy|mathematic|ecolog|natural history|environment",
     "Science & Nature", r"fiction|children|young adult"),
    (r"self[- ]help|psycholog|personal development|self-esteem|mental health|mindful|motivat|happiness|well-?being",
     "Self-help & Psychology", None),
    (r"philosoph|religio|spiritual|christian|bible|buddh|islam|theolog|ethics|occult", "Philosophy & Religion", r"horror"),
    (r"poetry|poems|\bplays?\b|drama|theatre|shakespeare", "Poetry & Drama", None),
    (r"graphic novel|comic|manga|cartoon", "Graphic Novels & Comics", None),
    (r"business|econom|financ|management|marketing|invest|entrepreneur|leadership|accounting", "Business & Economics", None),
    (r"cook|food|recipe|baking|wine|diet\b|lifestyle|craft|gardening|travel|fitness", "Cooking & Lifestyle", None),
]
_COMPILED_RULES = [(re.compile(p, re.I), g, re.compile(x, re.I) if x else None)
                   for p, g, x in CATEGORY_RULES]


def rule_genres(category_name: str) -> list[str]:
    """Genres déclenchés par les règles mots-clés pour un nom de catégorie."""
    found: list[str] = []
    for pat, genre, excl in _COMPILED_RULES:
        if pat.search(category_name) and not (excl and excl.search(category_name)):
            if genre not in found:
                found.append(genre)
    return found


def category_counts(df: pd.DataFrame) -> Counter:
    """Fréquence de chaque catégorie dans le catalogue (colonne ``categories`` = liste de noms)."""
    c: Counter = Counter()
    for cats in df["categories"]:
        c.update(list(cats) if cats is not None else [])
    return c


def build_category_map(counts: Counter, encoder: Encoder, blocks: dict, existing: pd.DataFrame | None = None,
                       min_sim: float = CATEGORY_MIN_SIM) -> pd.DataFrame:
    """Table de correspondance catégorie -> genre(s).

    Colonnes : category, n_books, rule_genres, emb_genre, emb_sim, genre_auto, genre_manual, genres_final.
    - ``genre_auto``   : règles si elles se déclenchent, sinon genre le plus proche par embeddings
                         (si cosinus >= min_sim, sinon vide : catégories génériques « Fiction », « Books »...).
    - ``genre_manual`` : colonne À REMPLIR à la main (relecture des 100 premières catégories) ;
                         séparateur « | ». Elle est conservée si le fichier est régénéré (``existing``).
    - ``genres_final`` : genre_manual s'il existe, sinon genre_auto.
    """
    cats = [c for c, _ in counts.most_common()]
    genre_labels = list(blocks["Genre"])
    g_emb = np.asarray(encoder.encode([blocks["Genre"][g] for g in genre_labels]))
    c_emb = np.asarray(encoder.encode(cats))
    sims = c_emb @ g_emb.T
    best = sims.argmax(1)

    rows = []
    for i, cat in enumerate(cats):
        rg = rule_genres(cat)
        emb_g, emb_s = genre_labels[best[i]], float(sims[i, best[i]])
        if rg:
            auto = rg
        elif emb_s >= min_sim:
            auto = [emb_g]
        else:
            auto = []
        rows.append({"category": cat, "n_books": counts[cat], "rule_genres": "|".join(rg),
                     "emb_genre": emb_g, "emb_sim": round(emb_s, 3),
                     "genre_auto": "|".join(auto), "genre_manual": ""})
    out = pd.DataFrame(rows)
    if existing is not None and "genre_manual" in existing:  # on garde le travail manuel déjà fait
        manual = existing.set_index("category")["genre_manual"].fillna("")
        out["genre_manual"] = out["category"].map(manual).fillna("")
    out["genres_final"] = np.where(out["genre_manual"].astype(str).str.strip() != "",
                                   out["genre_manual"], out["genre_auto"])
    return out


def load_category_map(path: Path | str = CATEGORY_MAP_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, keep_default_na=False)
    df["genres_final"] = np.where(df["genre_manual"].astype(str).str.strip() != "",
                                  df["genre_manual"], df["genre_auto"])
    return df


def assign_genres(df: pd.DataFrame, book_emb: np.ndarray, genre_labels: list[str], genre_emb: np.ndarray,
                  cat_map: pd.DataFrame, top_k: int = TOP_K["Genre"]) -> tuple[list[list[str]], dict]:
    """Genres de chaque livre.

    1. Union des genres de ses catégories (la table de correspondance de build_category_map).
    2. + « Children's » / « Young Adult » d'après la colonne ``audience`` de P1.
    3. Classement : nb de catégories qui soutiennent le genre, puis cosinus description <-> genre.
    4. Repli (aucune catégorie exploitable) : genre le plus proche de la description,
       restreint au domaine du livre (fiction / non-fiction).
    """
    cat2genres = {r.category: [g for g in str(r.genres_final).split("|") if g]
                  for r in cat_map.itertuples()}
    g_index = {g: i for i, g in enumerate(genre_labels)}
    sims = book_emb @ genre_emb.T  # (N, G) cosinus description <-> description du genre

    result, n_fallback = [], 0
    for i, row in enumerate(df.itertuples(index=False)):
        votes: Counter = Counter()
        for cat in (list(row.categories) if row.categories is not None else []):
            votes.update(cat2genres.get(cat, []))
        audience = getattr(row, "audience", None)
        if audience == "Children":
            votes["Children's"] += 1
        elif audience == "Young Adult":
            votes["Young Adult"] += 1

        # Genre le plus proche de la DESCRIPTION, cohérent avec le domaine fiction / non-fiction.
        # Il vote toujours (poids EMB_VOTE) : corrige les catégories commerciales trompeuses
        # (ex. Harry Potter classé « Sci Fi » par Book Depository).
        is_fic = bool(getattr(row, "is_fiction", True))
        allowed = [g for g in genre_labels
                   if (g in NEUTRAL_GENRES) or ((g in NONFICTION_GENRES) != is_fic)]
        emb_best = max(allowed, key=lambda g: sims[i, g_index[g]])
        if not votes:
            n_fallback += 1
        votes[emb_best] += EMB_VOTE
        ranked = sorted(votes, key=lambda g: (votes[g], sims[i, g_index[g]]), reverse=True)
        result.append(ranked[:top_k])
    stats = {"n_books": len(df), "n_fallback_embedding": n_fallback,
             "pct_fallback": round(100 * n_fallback / max(len(df), 1), 2)}
    return result, stats


# --------------------------------------------------------------------------- #
# 3. Zero-shot par embeddings (Themes / Style / Atmosphere)
# --------------------------------------------------------------------------- #
def score_block(book_emb: np.ndarray, label_emb: np.ndarray) -> np.ndarray:
    """Cosinus (N, L) entre description de chaque livre et phrase de chaque label (vecteurs normalisés)."""
    return book_emb @ label_emb.T


def zscore_columns(sims: np.ndarray) -> np.ndarray:
    """Normalise les scores par label : centrage par label + échelle COMMUNE à tout le bloc.

    Pourquoi centrer : certains labels ont un cosinus moyen élevé pour TOUS les livres (phrases génériques).
    Soustraire la moyenne du label revient à demander « ce livre est-il plus proche de ce label que le livre moyen ? ».
    Pourquoi une échelle commune (et non un écart-type par label) : diviser par l'écart-type de chaque label
    gonflait artificiellement les labels peu dispersés et produisait des labels aberrants (Gatsby -> « Freedom »).
    """
    centered = sims - sims.mean(0, keepdims=True)
    return centered / max(float(centered.std()), 1e-9)


def select_labels(z: np.ndarray, labels: list[str], top_k: int, z_min: float = Z_MIN,
                  min_keep: int = 1, group_of: dict[str, str] | None = None,
                  group_max: int | None = None) -> list[list[str]]:
    """Garde, pour chaque livre, les ``top_k`` meilleurs labels dont le z-score >= z_min.

    - ``min_keep`` : on garde toujours au moins ce nombre de labels (jamais de livre sans label).
    - ``group_of`` / ``group_max`` : limite le nombre de labels par famille (ex. 1 époque + 1 ambiance).
    """
    out = []
    for row in z:
        order = np.argsort(-row)
        chosen: list[int] = []
        per_group: Counter = Counter()
        for j in order:
            if len(chosen) >= top_k:
                break
            if group_of is not None and group_max is not None and per_group[group_of[labels[j]]] >= group_max:
                continue
            if row[j] < z_min and len(chosen) >= min_keep:
                break
            chosen.append(j)
            if group_of is not None:
                per_group[group_of[labels[j]]] += 1
        out.append([labels[j] for j in chosen])
    return out


def assign_labels(book_emb: np.ndarray, labels: list[str], label_emb: np.ndarray, block: str,
                  z_min: float = Z_MIN, top_k: int | None = None):
    """Zero-shot par embeddings pour un bloc. Retourne (labels par livre, matrice de cosinus brute)."""
    sims = score_block(book_emb, label_emb)
    z = zscore_columns(sims)
    group_of = group_max = None
    if block == "Atmosphere":  # 1 époque + 1 ambiance
        group_of = {l: ("era" if l in ERA_LABELS else "mood") for l in labels}
        group_max = 1
    chosen = select_labels(z, labels, top_k or TOP_K[block], z_min, 1, group_of, group_max)
    return chosen, sims


def threshold_report(sims: np.ndarray, labels: list[str], block: str,
                     thresholds: Iterable[float] = (0.0, 0.5, 1.0, 1.5, 2.0)) -> pd.DataFrame:
    """Aide à CALIBRER le seuil z_min : nb moyen de labels / livre et % de livres sans label au seuil."""
    z = zscore_columns(sims)
    rows = []
    for t in thresholds:
        chosen, _ = assign_labels_from_z(z, labels, block, t)
        n_pass = (z >= t).sum(1)
        rows.append({"z_min": t, "mean_labels_kept": round(float(np.mean([len(c) for c in chosen])), 2),
                     "pct_books_no_label_above_threshold": round(float((n_pass == 0).mean() * 100), 1)})
    return pd.DataFrame(rows)


def assign_labels_from_z(z, labels, block, z_min):
    group_of = group_max = None
    if block == "Atmosphere":
        group_of = {l: ("era" if l in ERA_LABELS else "mood") for l in labels}
        group_max = 1
    return select_labels(z, labels, TOP_K[block], z_min, 1, group_of, group_max), z


def build_label_scores(sims_by_block: dict[str, tuple[list[str], np.ndarray]], decimals: int = 3) -> list[str]:
    """label_scores par livre = chaîne JSON {bloc: {label: cosinus}} (contrat : dict reconverti par load_books)."""
    n = next(iter(sims_by_block.values()))[1].shape[0]
    out = []
    for i in range(n):
        out.append(json.dumps({b: {l: round(float(s[i, j]), decimals) for j, l in enumerate(labels)}
                               for b, (labels, s) in sims_by_block.items()}))
    return out


# --------------------------------------------------------------------------- #
# 4. Mots-clés (méthode type KeyBERT)
# --------------------------------------------------------------------------- #
def _words(phrase: str) -> set[str]:
    return set(phrase.split())


def mmr_select(cand_emb: np.ndarray, doc_emb: np.ndarray, phrases: list[str], top_n: int = 5,
               diversity: float = 0.5) -> list[str]:
    """Maximal Marginal Relevance : pertinence au document vs redondance avec les mots-clés déjà choisis.

    On écarte aussi un candidat inclus dans un mot-clé déjà choisi (ou l'inverse) :
    « wizard » vs « boy wizard ».
    """
    if len(phrases) == 0:
        return []
    rel = cand_emb @ doc_emb
    chosen = [int(rel.argmax())]
    while len(chosen) < min(top_n, len(phrases)):
        red = (cand_emb @ cand_emb[chosen].T).max(1)
        score = (1 - diversity) * rel - diversity * red
        for j in chosen:
            score[j] = -np.inf
        for j in range(len(phrases)):  # inclusion lexicale = redondance
            if any(_words(phrases[j]) <= _words(phrases[k]) or _words(phrases[k]) <= _words(phrases[j])
                   for k in chosen):
                score[j] = -np.inf
        if np.isinf(score).all():
            break
        chosen.append(int(score.argmax()))
    return [phrases[j] for j in chosen]


def extract_keywords(df: pd.DataFrame, book_emb: np.ndarray, encoder: Encoder, top_n: int = 5,
                     n_candidates: int = 15, diversity: float = 0.5, ngram_range=(1, 2),
                     text_col: str = "description") -> list[list[str]]:
    """Mots-clés par livre, selon le principe de KeyBERT (cosinus candidat <-> document + MMR).

    Pourquoi pas KeyBERT directement sur 12 000 livres ? KeyBERT encode TOUT le vocabulaire
    (uni+bigrammes) de la collection : plusieurs centaines de milliers de phrases, trop lent
    sur CPU. Ici on pré-filtre les ``n_candidates`` meilleurs candidats par livre avec TF-IDF
    (rapide, creux), on n'encode que l'union dédupliquée, puis on réordonne par embeddings + MMR.
    Même algorithme, même modèle, quelques minutes au lieu de plusieurs heures.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    docs = df[text_col].fillna("").astype(str).tolist()
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    stop = sorted(set(ENGLISH_STOP_WORDS) | EXTRA_STOP)
    tfidf = TfidfVectorizer(ngram_range=ngram_range, stop_words=stop, lowercase=True,
                            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z-]{2,}\b",
                            min_df=2, max_df=0.5, sublinear_tf=True, dtype=np.float32)
    X = tfidf.fit_transform(docs).tocsr()
    vocab = np.array(tfidf.get_feature_names_out())

    authors = df["authors_str"].fillna("").astype(str) if "authors_str" in df else pd.Series([""] * len(df))
    author_words = [{w for w in re.findall(r"[a-z]{3,}", a.lower())} for a in authors]

    # 1. top-n_candidates TF-IDF par livre
    cand_ids: list[np.ndarray] = []
    for i in range(X.shape[0]):
        start, end = X.indptr[i], X.indptr[i + 1]
        idx, val = X.indices[start:end], X.data[start:end]
        top = idx[np.argsort(-val)]
        # on retire les candidats qui reprennent le nom d'un auteur du livre (« scott fitzgerald »)
        if author_words[i]:
            top = np.array([j for j in top if not (set(vocab[j].split()) & author_words[i])], dtype=int)
        cand_ids.append(top[:n_candidates])

    # 2. encodage de l'union dédupliquée des candidats
    union = np.unique(np.concatenate(cand_ids)) if cand_ids else np.array([], dtype=int)
    pos = {int(j): k for k, j in enumerate(union)}
    union_emb = np.asarray(encoder.encode(vocab[union].tolist())) if len(union) else np.zeros((0, 1))

    # 3. rerank embeddings + MMR
    out = []
    for i, ids in enumerate(cand_ids):
        if len(ids) == 0:
            out.append([])
            continue
        cand_emb = union_emb[[pos[int(j)] for j in ids]]
        out.append(mmr_select(cand_emb, book_emb[i], vocab[ids].tolist(), top_n, diversity))
    return out


def extract_keywords_keybert_lib(texts: list[str], model_name: str = DEFAULT_MODEL, top_n: int = 5):
    """Version avec la bibliothèque KeyBERT (à utiliser sur un ÉCHANTILLON pour comparer, ex. 200 livres)."""
    from keybert import KeyBERT
    kw = KeyBERT(model=model_name)
    res = kw.extract_keywords(texts, keyphrase_ngram_range=(1, 2), stop_words="english",
                              use_mmr=True, diversity=0.5, top_n=top_n)
    return [[k for k, _ in r] for r in res] if res and isinstance(res[0], list) else [[k for k, _ in res]]


# --------------------------------------------------------------------------- #
# 5. Pipeline complet + export
# --------------------------------------------------------------------------- #
CONTRACT_NEW_COLS = ["genres", "themes", "styles", "atmospheres", "keywords", "label_scores"]


def enrich_catalogue(df: pd.DataFrame, encoder: Encoder, blocks: dict | None = None,
                     cat_map: pd.DataFrame | None = None, book_emb: np.ndarray | None = None,
                     z_min: float = Z_MIN, with_keywords: bool = True, verbose: bool = True):
    """Enrichit ``df`` (books_clean ou books_sample). Retourne (df enrichi, rapport dict).

    Les colonnes de P2 éventuellement déjà présentes mais vides (cas de books_sample.parquet) sont écrasées.
    """
    blocks = blocks or load_blocks()
    df = df.drop(columns=[c for c in CONTRACT_NEW_COLS if c in df.columns]).reset_index(drop=True)
    log = print if verbose else (lambda *a, **k: None)

    log("[1/5] Encodage des livres et des phrases de labels")
    if book_emb is None:
        book_emb = np.asarray(encoder.encode(df["embed_text"].fillna("").tolist()))
    enc_blocks = encode_blocks(encoder, blocks)

    log("[2/5] Genres (catégories -> genres, repli embeddings)")
    if cat_map is None:
        cat_map = build_category_map(category_counts(df), encoder, blocks)
    g_labels, g_emb = enc_blocks["Genre"]
    genres, g_stats = assign_genres(df, book_emb, g_labels, g_emb, cat_map)

    log("[3/5] Zero-shot Themes / Style / Atmosphere")
    sims_by_block = {"Genre": (g_labels, score_block(book_emb, g_emb))}
    assigned = {}
    for block in ["Themes", "Style", "Atmosphere"]:
        labels, emb = enc_blocks[block]
        assigned[block], sims = assign_labels(book_emb, labels, emb, block, z_min)
        sims_by_block[block] = (labels, sims)

    log("[4/5] Mots-clés")
    keywords = extract_keywords(df, book_emb, encoder) if with_keywords else [[] for _ in range(len(df))]

    log("[5/5] Assemblage")
    out = df.copy()
    out["genres"], out["themes"] = genres, assigned["Themes"]
    out["styles"], out["atmospheres"] = assigned["Style"], assigned["Atmosphere"]
    out["keywords"] = keywords
    out["label_scores"] = build_label_scores(sims_by_block)

    report = {"n_books": len(out), "z_min": z_min, "genres": g_stats,
              "mean_labels": {c: round(float(out[c].map(len).mean()), 2)
                              for c in ["genres", "themes", "styles", "atmospheres", "keywords"]}}
    return out, report


def save_enriched(df: pd.DataFrame, path: Path | str) -> None:
    """Écrit le parquet en normalisant les types (listes Python, label_scores en chaîne JSON)."""
    df = df.copy()
    for c in ["authors", "categories", "genres", "themes", "styles", "atmospheres", "keywords"]:
        if c in df.columns:
            df[c] = df[c].map(lambda v: [] if v is None else [str(x) for x in list(v)])
    if "label_scores" in df.columns:
        df["label_scores"] = df["label_scores"].map(lambda v: v if isinstance(v, str) else json.dumps(v))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


# --------------------------------------------------------------------------- #
# 6. Validation sur les livres annotés à la main
# --------------------------------------------------------------------------- #
def make_annotation_sheet(df: pd.DataFrame, n: int = 60, seed: int = 42, n_nonfiction: int = 15) -> pd.DataFrame:
    """Feuille d'annotation (SANS les prédictions, pour ne pas biaiser l'annotateur).

    Remplir gold_genres (1-3 labels) et gold_themes (2-3 labels) avec les labels de blocks.json, séparés par « | ».
    """
    fic = df[df["is_fiction"]].sample(n - n_nonfiction, random_state=seed)
    non = df[~df["is_fiction"]].sample(min(n_nonfiction, int((~df["is_fiction"]).sum())), random_state=seed)
    sheet = pd.concat([fic, non]).sample(frac=1, random_state=seed)
    sheet = sheet[["book_id", "title", "authors_str", "description"]].copy()
    sheet["description"] = sheet["description"].str.slice(0, 500)
    sheet["gold_genres"], sheet["gold_themes"], sheet["annotator"] = "", "", ""
    return sheet


def evaluate_against_gold(enriched: pd.DataFrame, gold_path: Path | str) -> dict:
    """Compare les labels prédits à l'annotation manuelle (colonnes gold_genres / gold_themes)."""
    gold = pd.read_csv(gold_path, keep_default_na=False, dtype={"book_id": str})
    gold = gold[(gold["gold_genres"].str.strip() != "") | (gold["gold_themes"].str.strip() != "")]
    enriched = enriched.assign(book_id=enriched["book_id"].astype(str))
    merged = gold.merge(enriched[["book_id", "genres", "themes"]], on="book_id", how="inner")

    def metrics(pred_col: str, gold_col: str) -> dict:
        prec, rec, hit1, hit_any = [], [], [], []
        for _, r in merged.iterrows():
            g = {x.strip() for x in str(r[gold_col]).split("|") if x.strip()}
            p = list(r[pred_col])
            if not g or not p:
                continue
            inter = len(g & set(p))
            prec.append(inter / len(p))
            rec.append(inter / len(g))
            hit1.append(p[0] in g)
            hit_any.append(inter > 0)
        n = max(len(prec), 1)
        return {"n": len(prec), "precision@k": round(sum(prec) / n, 3), "recall": round(sum(rec) / n, 3),
                "top1_in_gold": round(sum(hit1) / n, 3), "at_least_one_match": round(sum(hit_any) / n, 3)}

    return {"genres": metrics("genres", "gold_genres"), "themes": metrics("themes", "gold_themes")}
