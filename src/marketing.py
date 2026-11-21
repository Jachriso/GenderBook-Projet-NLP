"""
Retrait du texte marketing des descriptions (citations de presse, bandeaux commerciaux).

Pourquoi : ~7 % des descriptions Book Depository commencent par des critiques
    "A hair-raising debut..." --Mary Kubica, New York Times bestselling author of The Good Girl
ou des bandeaux (« A New York Times Bestseller », « NOW A MAJOR MOTION PICTURE »).
Ces phrases parlent du SUCCÈS du livre, pas de son CONTENU : elles rapprocheraient
artificiellement tous les best-sellers entre eux dans l'espace d'embedding.

La difficulté est de savoir OÙ S'ARRÊTE L'ATTRIBUTION qui suit une citation.
Dans « --People Meghan doesn't know Agatha », « People » est un magazine mais
« Meghan » est déjà le début de l'intrigue. Une regex « suite de mots en
majuscule » ne peut pas faire la différence. On utilise donc des connaissances
tirées du dataset lui-même :
  * une liste de PUBLICATIONS de presse connues (People, Booklist, Kirkus…) ;
  * la liste des NOMS D'AUTEURS du dataset (authors.csv) : « Stephen King » ;
  * la liste des TITRES du dataset, pour savoir où s'arrête
    « author of The Statistical Probability of Love at First Sight ».
Sans ces listes (ex. dans les tests), on retombe sur des heuristiques prudentes
qui préfèrent laisser quelques mots de marketing plutôt que couper du contenu.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Iterable

_MULTI_SPACE_RE = re.compile(r"\s+")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.;:!?])")

# ---------------------------------------------------------------------------
# Normalisation commune (noms, titres)
# ---------------------------------------------------------------------------

def norm_words(text: str) -> str:
    """Minuscules ASCII, sans apostrophes / points / tirets, espaces simples.

    "J. K. Rowling" -> "j k rowling" ; "Cat's Eye" -> "cats eye".
    """
    t = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    t = re.sub(r"['’.\-]", "", t)
    t = re.sub(r"&", " and ", t)
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return t.strip()


# ---------------------------------------------------------------------------
# Connaissances : publications de presse fréquemment citées
# ---------------------------------------------------------------------------
PUBLICATIONS = sorted({
    "people", "people magazine", "booklist", "kirkus", "kirkus reviews", "publishers weekly",
    "library journal", "school library journal", "the horn book", "horn book", "voya",
    "bookpage", "shelf awareness", "booksellers", "the bookseller", "rt book reviews",
    "romantic times", "entertainment weekly", "ew", "usa today", "time", "time magazine",
    "newsweek", "the new york times", "new york times", "the new york times book review",
    "new york times book review", "the washington post", "washington post", "the wall street journal",
    "wall street journal", "los angeles times", "la times", "chicago tribune", "boston globe",
    "the boston globe", "san francisco chronicle", "associated press", "npr", "the atlantic",
    "the new yorker", "new yorker", "vogue", "elle", "glamour", "cosmopolitan", "marie claire",
    "harper's bazaar", "harpers bazaar", "o, the oprah magazine", "o the oprah magazine",
    "oprah magazine", "o magazine", "huffington post", "huffpost", "buzzfeed", "bustle",
    "refinery29", "vanity fair", "esquire", "rolling stone", "the economist", "economist",
    "financial times", "the guardian", "guardian", "the observer", "observer", "the times",
    "the sunday times", "sunday times", "the independent", "independent", "independent on sunday",
    "the telegraph", "telegraph", "daily telegraph", "sunday telegraph", "daily mail",
    "mail on sunday", "daily express", "sunday express", "evening standard", "the spectator",
    "spectator", "literary review", "times literary supplement", "tls", "the scotsman",
    "scotsman", "irish times", "the irish times", "heat", "grazia", "stylist", "red",
    "good housekeeping", "woman & home", "woman and home", "prima", "closer", "bella",
    "metro", "the sun", "daily mirror", "mirror", "new statesman", "the times uk",
    "kirkus reviews starred review", "booklist starred review", "publishers weekly starred review",
    "the new york times bestselling", "goodreads", "amazon", "audible", "popsugar", "newsday",
    "sfrevu", "sf site", "locus", "tor.com", "io9", "slate", "salon", "the daily beast",
    "minneapolis star tribune", "star tribune", "seattle times", "miami herald", "dallas morning news",
    "the oprah magazine", "o, the oprah magazine", "parade", "us weekly", "in style", "instyle",
    "real simple", "southern living", "family circle", "redbook", "self", "shape", "marie claire uk",
    "the sunday express", "the daily mail", "the express", "the mail on sunday", "the sun",
    "the new york review of books", "new york review of books", "the new york times review of books",
    "the los angeles review of books", "the london review of books", "london review of books",
}, key=len, reverse=True)
# particules de noms propres en minuscule : "Alain de Botton", "Ludwig van Beethoven"
_NAME_PARTICLES = {"de", "van", "von", "der", "den", "da", "di", "du", "le", "la", "del", "bin", "al", "st"}

# Mots qui, en 2e position après un tiret d'attribution, commencent presque
# toujours la VRAIE description : "--PopSugar In this entrancing novel…",
# "--Newsday Winner of the Whitbread Prize…". On ne les compte pas dans le nom.
_SENTENCE_STARTERS = {
    "the", "a", "an", "in", "on", "at", "as", "it", "he", "she", "they", "we", "i", "his", "her",
    "their", "this", "that", "these", "those", "when", "while", "after", "before", "from", "for",
    "with", "winner", "now", "soon", "praise", "once", "one", "two", "set", "here", "there", "what",
    "why", "how", "meet", "welcome", "discover", "like", "if", "but", "and", "since", "during",
    "every", "some", "born", "growing", "years", "book", "volume", "part", "featuring", "includes",
    "nothing", "everything", "everyone", "no", "not", "my", "our", "your", "its", "after", "by",
    "shortlisted", "longlisted", "new", "original", "first", "last", "only", "all", "back",
    "though", "although", "even", "yet", "still", "despite", "having", "until", "where", "who",
    "once", "so", "because", "although", "whether", "unlike", "beneath", "inside", "across",
}

# ---------------------------------------------------------------------------
# Expressions régulières
# ---------------------------------------------------------------------------
_Q_OPEN, _Q_CLOSE = "\"“«", "\"”»"
_DASH = r"(?:--|—|–|―|-{3}|\s-\s)"
# citation entre guillemets suivie d'un tiret d'attribution
_BLURB_START_RE = re.compile(rf"[{_Q_OPEN}][^{_Q_CLOSE}]{{2,600}}[{_Q_CLOSE}]\s*{_DASH}\s*")
# citation entre parenthèses : ("remarkable"--Associated Press)
_PAREN_BLURB_RE = re.compile(
    rf"\(\s*[{_Q_OPEN}][^{_Q_CLOSE}]{{1,250}}[{_Q_CLOSE}]\s*[-–—―]{{1,3}}[^)]{{1,80}}\)"
)
# précision après une publication : "(starred review)", ", starred review"
_REVIEW_SUFFIX_RE = re.compile(
    r"\s*(?:\(\s*starred review\s*\)|,?\s*starred review\b|\(\s*(?:uk|us)\s*\)|\s+review\b)",
    flags=re.IGNORECASE,
)
# rôle après un nom : ", New York Times and USA Today bestselling author of"
_ROLE_RE = re.compile(
    r"\s*,\s*(?:the\s+)?(?:(?!(?:from|by|in|on|with)\b)[#\w'’.&-]+\s+){0,8}?"
    r"(?:author|writer|novelist|co-?creator|creator|co-?star|editor|host|co-?author|illustrator|"
    r"founder|director|columnist|journalist|critic)\b(?:\s+of\b)?",
    flags=re.IGNORECASE,
)
# bandeau commercial EN DÉBUT de texte (casse quelconque)
_LEADING_SALES_RE = re.compile(
    r"^\s*(?:(?:an?|the)\s+)?"
    r"(?:(?:instant|#\s?1|no\.?\s?1|number one|international(?:ly)?|national|global|"
    r"sunday times|new york times|usa today|wall street journal|indie|times|top ten|top 10|"
    r"million[- ]copy|smash|runaway|beloved)\s*(?:and\s+)?)+"
    r"best-?sell(?:er|ing)\w*(?:\s+(?:novel|book|series|author|memoir|thriller|phenomenon))?"
    r"\s*[!.:;,\-–—]*",
    flags=re.IGNORECASE,
)
_LEADING_PROMO_RE = re.compile(
    r"^\s*(?:"
    r"(?:now|soon to be)\s+an?\s+(?:major\s+)?(?:new\s+)?(?:motion picture|film|movie|feature film|"
    r"netflix (?:original )?(?:series|film)|tv series|hbo (?:max )?(?:series|original)|"
    r"amazon (?:prime )?(?:video )?series|television series|limited series)[^.!]{0,80}[.!]?"
    r"|(?:an?\s+)?(?:oprah'?s|reese'?s|good morning america|today show|jenna'?s|read with jenna|"
    r"richard (?:and|&) judy)\s+book club\s+(?:pick|selection|choice)[!.]?"
    r"|(?:winner of|shortlisted for|longlisted for)\s+the\s+[^.!]{3,120}(?:award|prize|medal)s?[.!]?"
    r")",
    flags=re.IGNORECASE,
)
# bandeau en MAJUSCULES (≥ 3 mots sans minuscule) contenant un argument de vente
_CAPS_HEADLINE_RE = re.compile(
    r"(?<![\w'’])(?:[A-Z0-9#'’!?,.&:;\-]+\s+){2,}[A-Z0-9#'’!?,.&:;\-]+(?![\w'’])"
)
_SALES_WORDS_RE = re.compile(
    r"BESTSELL|BEST-SELL|NEW YORK TIMES|SUNDAY TIMES|USA TODAY|#1|NUMBER ONE|"
    r"WINNER|AWARD|PRIZE|MAJOR (?:MOTION PICTURE|NETFLIX|TV)|NOW A |SOON TO BE|"
    r"OVER \d|MILLION COPIES|INSTANT|EDITION|BOOK CLUB"
)
# en-tête d'une section de critiques : "Praise for I Am Zlatan", "Praise for the Temeraire series:"
_PRAISE_HEADER_RE = re.compile(
    rf"\b(?:advance\s+|critical\s+)?(?:praise|acclaim)\s+for\b[^.!?{_Q_OPEN}]{{0,120}}?(?::|(?=\s*[{_Q_OPEN}])|$)",
    flags=re.IGNORECASE,
)
_TOKEN_RE = re.compile(r"\S+")
# petits mots qui peuvent commencer la VRAIE phrase juste après un bandeau en majuscules
_CAPS_SENTENCE_STARTERS = {"A", "AN", "THE", "I", "IN", "ON", "AT", "AS", "IT", "HE", "SHE", "WE", "THEY"}


def _followed_by_lowercase(text: str, end: int) -> bool:
    """True si le texte après `end` commence par un mot en minuscule (= on couperait une phrase)."""
    # on saute virgules et tirets : "Winner of the Pulitzer Prize--a powerful love story"
    # ou "…Award, the #1 bestseller from…" sont des appositions, pas une nouvelle phrase
    rest = text[end:].lstrip(" ,;-–—")
    return bool(rest) and rest[0].islower()


def _sub_leading_if_standalone(regex: re.Pattern, text: str) -> str:
    """Retire le bandeau en début de texte SEULEMENT s'il forme un segment autonome.

    "#1 NEW YORK TIMES BESTSELLER - A pair of sisters…" -> retiré ;
    "The #1 New York Times bestselling debut novel that…" -> conservé (la suite
    "debut novel that…" n'aurait plus de sens).
    """
    m = regex.match(text)
    if not m or _followed_by_lowercase(text, m.end()):
        return text
    return text[m.end():]
_SMALL_WORDS = {"of", "the", "a", "an", "and", "in", "on", "at", "to", "for", "from", "with", "or", "by", "&"}
_INITIAL_RE = re.compile(r"^[A-Z]\.?(?:[A-Z]\.?)*,?$")


def _is_cap(tok: str) -> bool:
    return tok[:1].isupper() or tok[:1] in "#'’"


class MarketingStripper:
    """Retire citations de presse et bandeaux commerciaux d'une description.

    author_names / titles : listes issues du dataset (authors.csv, titres),
    utilisées pour savoir où s'arrête une attribution. Optionnelles.
    """

    def __init__(self, author_names: Iterable[str] = (), titles: Iterable[str] = ()):
        self.names = {n for n in (norm_words(a) for a in author_names) if 2 <= len(n.split()) <= 5}
        self.titles = {t for t in (norm_words(x) for x in titles) if 1 <= len(t.split()) <= 12}

    # -- résolution de l'attribution ------------------------------------------
    @staticmethod
    def _publication_len(sub: str) -> int:
        """Longueur de la publication de presse connue en tête de sub (0 sinon)."""
        low = sub.lower()
        for pub in PUBLICATIONS:
            if low.startswith(pub) and (
                len(sub) == len(pub)
                or not sub[len(pub)].isalnum()
                # publication collée au mot suivant faute d'espace : "--TLSThe war raged…"
                or (sub[len(pub)].isupper() and sub[len(pub) - 1].isalpha()
                    and sub[:len(pub)].isupper() or (sub[len(pub)].isupper() and sub[len(pub) - 1].islower()))
            ):
                return len(pub)
        return 0

    def _source_end(self, sub: str) -> int:
        """Longueur de la « source » en tête de sub : publication, auteur connu ou nom probable."""
        pub_len = self._publication_len(sub)
        if pub_len:
            return pub_len
        toks = list(_TOKEN_RE.finditer(sub[:160]))[:6]
        # nom d'auteur connu (le plus long d'abord)
        if self.names:
            for n in range(min(5, len(toks)), 1, -1):
                if not _is_cap(toks[0].group()) or not all(
                        _is_cap(t.group()) or t.group().lower() in _NAME_PARTICLES for t in toks[:n]):
                    continue
                span = sub[toks[0].start():toks[n - 1].end()].rstrip(",;:")
                if norm_words(span) in self.names:
                    return toks[0].start() + len(span)
        # repli prudent : "Prénom [Initiale] Nom" (2 ou 3 mots en majuscule), sinon 1 mot
        caps = 0
        for t in toks:
            if not _is_cap(t.group()):
                break
            caps += 1
        if caps == 0:
            return 0
        n = 1
        if caps >= 2 and toks[1].group().strip(",;:").lower() not in _SENTENCE_STARTERS:
            n = 3 if caps >= 3 and _INITIAL_RE.match(toks[1].group()) else 2
        if toks[n - 1].group().endswith(","):
            span_end = toks[n - 1].end() - 1
        else:
            span_end = toks[n - 1].end()
        return span_end

    def _title_end(self, sub: str) -> int:
        """Longueur du (ou des) titre(s) d'œuvre en tête de sub, après « author of »."""
        end = 0
        for _ in range(3):  # "author of A, B and C"
            lead = re.match(r"the\s+(?=[A-Z])", sub[end:])  # "author of the Pip & Flinx series"
            if lead:
                end += lead.end()
            rest = sub[end:]
            toks = list(_TOKEN_RE.finditer(rest[:200]))[:12]
            if not toks or not _is_cap(toks[0].group().lstrip("\"“'‘*_")):
                break
            # (1) titre connu du dataset, le plus long possible
            known, known_len = 0, 0
            if self.titles:
                for k in range(len(toks), 0, -1):
                    span = rest[toks[0].start():toks[k - 1].end()].rstrip(",;:.!?")
                    if norm_words(span) in self.titles:
                        known, known_len = toks[0].start() + len(span), k
                        break
            # (2) heuristique : mots en majuscule / petits mots, en s'arrêtant avant un
            #     mot en majuscule suivi d'un mot en minuscule (= début de la vraie phrase)
            cap_tokens: list[re.Match] = []
            for i, t in enumerate(toks[:6]):
                w = t.group()
                nxt = toks[i + 1].group() if i + 1 < len(toks) else ""
                if i > 0 and _is_cap(w) and nxt[:1].islower() and nxt.lower() not in _SMALL_WORDS:
                    break
                if not (_is_cap(w) or w.lower() in _SMALL_WORDS):
                    break
                if _is_cap(w):
                    cap_tokens.append(t)
            # un mot-outil en majuscule en fin de titre ouvre en fait la phrase suivante :
            # "author of Tiny Beautiful Things On the eve of…" -> "On" n'appartient pas au titre
            while len(cap_tokens) > 1 and cap_tokens[-1].group().lower() in _SENTENCE_STARTERS:
                cap_tokens.pop()
            heuristic = cap_tokens[-1].end() if cap_tokens else 0
            # un titre connu d'UN seul mot ("Sweet") ne bat pas une heuristique plus
            # longue ("Sweet Evil") : beaucoup de mots courants sont aussi des titres
            matched = known if known and (known_len >= 2 or heuristic <= known) else heuristic
            if not matched:
                break
            end += matched
            suffix = re.match(r"\s+(?:series|trilogy|saga|novels|books|mysteries|chronicles|quartet|"
                              r"cycle|sequence)\b", sub[end:], flags=re.IGNORECASE)
            if suffix:
                end += suffix.end()
            sep = re.match(r"\s*(?:,\s*and|,|and|&)\s+", sub[end:])
            if not sep:
                break
            end += sep.end()
        return end

    def _attribution_end(self, text: str, pos: int) -> int:
        """Position de fin de l'attribution qui commence à `pos` (après le tiret)."""
        sub = text[pos:pos + 400]
        end = self._source_end(sub)
        m = _REVIEW_SUFFIX_RE.match(sub, end)
        if m:
            end = m.end()
        m = _ROLE_RE.match(sub, end)
        if m:
            # ", New York Times and USA Today bestselling author of The Good Girl"
            end = m.end()
            if m.group(0).lower().rstrip().endswith("of"):
                end += self._title_end(sub[end:].lstrip()) + (len(sub[end:]) - len(sub[end:].lstrip()))
        else:
            # "Janet Maslin, The New York Times" : nom puis publication
            m = re.match(r"\s*,\s*", sub[end:])
            if m:
                after = end + m.end()
                pub_len = self._publication_len(sub[after:])
                if pub_len:
                    end = after + pub_len
        return pos + end

    # -- étapes de nettoyage ---------------------------------------------------
    def _strip_blurbs(self, text: str) -> str:
        pos = 0
        while True:
            m = _BLURB_START_RE.search(text, pos)
            if not m:
                return text
            end = self._attribution_end(text, m.end())
            if end == m.end():
                # aucune source après le tiret (mot en minuscule) : ce n'est pas une
                # critique de presse mais une citation DANS le récit -> on n'y touche pas
                # ex. "a Turkish inventor known as Hoja--\"master\"--a man who is his double"
                pos = m.end()
                continue
            text = text[:m.start()] + " " + text[end:].lstrip(" ,;.")
            pos = m.start()

    @staticmethod
    def _drop_sales_caps(match: re.Match) -> str:
        """Retire un bandeau en MAJUSCULES s'il contient un argument de vente.

        Si la suite commence par une minuscule ("NEW YORK TIMES BESTSELLER A superb
        love story"), le dernier mot du segment ("A") est en fait le début de la
        vraie phrase : on le rend au texte. Si ce n'est pas le cas, on garde tout
        (on préfère laisser du marketing que couper une phrase).
        """
        seg = match.group(0)
        if not _SALES_WORDS_RE.search(seg):
            return seg
        if not _followed_by_lowercase(match.string, match.end()):
            return " "
        toks = seg.split()
        if toks and toks[-1].strip(",;:") in _CAPS_SENTENCE_STARTERS and len(toks) >= 4:
            head = " ".join(toks[:-1])
            if _SALES_WORDS_RE.search(head):
                return " " + toks[-1].capitalize() if toks[-1] != "I" else " I"
        return seg

    def __call__(self, text: str) -> str:
        if not text:
            return text
        for _ in range(8):  # bandeaux et citations peuvent s'enchaîner
            before = text
            # 1) bandeaux en MAJUSCULES d'abord (sinon la regex en casse mixte n'en
            #    retirerait que le début : "THE INSTANT NEW YORK TIMES BESTSELLING | DEBUT…")
            text = _CAPS_HEADLINE_RE.sub(self._drop_sales_caps, text).lstrip()
            text = _sub_leading_if_standalone(_LEADING_SALES_RE, text).lstrip(" ,;:.!-–—")
            text = _sub_leading_if_standalone(_LEADING_PROMO_RE, text).lstrip(" ,;:.!-–—")
            text = _PAREN_BLURB_RE.sub(" ", text)
            text = self._strip_blurbs(text)
            text = _PRAISE_HEADER_RE.sub(" ", text)
            text = _MULTI_SPACE_RE.sub(" ", text).lstrip(" ,;:.!-–—")
            if text == before:
                break
        text = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
        return text.strip(" ,;:-–—")


# Instance par défaut, sans listes (heuristiques prudentes)
_DEFAULT = MarketingStripper()


def strip_marketing(text: str, stripper: MarketingStripper | None = None) -> str:
    """Raccourci : retire le texte marketing avec `stripper` (ou l'instance par défaut)."""
    return (stripper or _DEFAULT)(text)
