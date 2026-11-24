"""
Removal of marketing text from the descriptions (press quotes, commercial banners).

Why: ~7% of the Book Depository descriptions start with reviews
    "A hair-raising debut..." --Mary Kubica, New York Times bestselling author of The Good Girl
or banners ("A New York Times Bestseller", "NOW A MAJOR MOTION PICTURE").
These sentences talk about the SUCCESS of the book, not its CONTENT: they would
artificially bring all best-sellers closer together in the embedding space.

The difficulty is to know WHERE THE ATTRIBUTION that follows a quote ENDS.
In "--People Meghan doesn't know Agatha", "People" is a magazine but
"Meghan" is already the start of the plot. A "sequence of capitalised
words" regex cannot tell the difference. We therefore use knowledge
taken from the dataset itself:
  * a list of well-known press PUBLICATIONS (People, Booklist, Kirkus…);
  * the list of AUTHOR NAMES of the dataset (authors.csv): "Stephen King";
  * the list of TITLES of the dataset, to know where
    "author of The Statistical Probability of Love at First Sight" ends.
Without these lists (e.g. in the tests), we fall back on cautious heuristics
that prefer leaving a few marketing words rather than cutting content.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Iterable

_MULTI_SPACE_RE = re.compile(r"\s+")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.;:!?])")

# ---------------------------------------------------------------------------
# Shared normalisation (names, titles)
# ---------------------------------------------------------------------------

def norm_words(text: str) -> str:
    """ASCII lowercase, without apostrophes / dots / hyphens, single spaces.

    "J. K. Rowling" -> "j k rowling"; "Cat's Eye" -> "cats eye".
    """
    t = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    t = re.sub(r"['’.\-]", "", t)
    t = re.sub(r"&", " and ", t)
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return t.strip()


# ---------------------------------------------------------------------------
# Knowledge: frequently quoted press publications
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
# lowercase particles of proper names: "Alain de Botton", "Ludwig van Beethoven"
_NAME_PARTICLES = {"de", "van", "von", "der", "den", "da", "di", "du", "le", "la", "del", "bin", "al", "st"}

# Words that, in 2nd position after an attribution dash, almost always start
# the REAL description: "--PopSugar In this entrancing novel…",
# "--Newsday Winner of the Whitbread Prize…". They are not counted in the name.
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
# Regular expressions
# ---------------------------------------------------------------------------
_Q_OPEN, _Q_CLOSE = "\"“«", "\"”»"
_DASH = r"(?:--|—|–|―|-{3}|\s-\s)"
# quote in quotation marks followed by an attribution dash
_BLURB_START_RE = re.compile(rf"[{_Q_OPEN}][^{_Q_CLOSE}]{{2,600}}[{_Q_CLOSE}]\s*{_DASH}\s*")
# quote in parentheses: ("remarkable"--Associated Press)
_PAREN_BLURB_RE = re.compile(
    rf"\(\s*[{_Q_OPEN}][^{_Q_CLOSE}]{{1,250}}[{_Q_CLOSE}]\s*[-–—―]{{1,3}}[^)]{{1,80}}\)"
)
# detail after a publication: "(starred review)", ", starred review"
_REVIEW_SUFFIX_RE = re.compile(
    r"\s*(?:\(\s*starred review\s*\)|,?\s*starred review\b|\(\s*(?:uk|us)\s*\)|\s+review\b)",
    flags=re.IGNORECASE,
)
# role after a name: ", New York Times and USA Today bestselling author of"
_ROLE_RE = re.compile(
    r"\s*,\s*(?:the\s+)?(?:(?!(?:from|by|in|on|with)\b)[#\w'’.&-]+\s+){0,8}?"
    r"(?:author|writer|novelist|co-?creator|creator|co-?star|editor|host|co-?author|illustrator|"
    r"founder|director|columnist|journalist|critic)\b(?:\s+of\b)?",
    flags=re.IGNORECASE,
)
# commercial banner at the START of the text (any case)
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
# banner in CAPITALS (≥ 3 words without lowercase) containing a sales pitch
_CAPS_HEADLINE_RE = re.compile(
    r"(?<![\w'’])(?:[A-Z0-9#'’!?,.&:;\-]+\s+){2,}[A-Z0-9#'’!?,.&:;\-]+(?![\w'’])"
)
_SALES_WORDS_RE = re.compile(
    r"BESTSELL|BEST-SELL|NEW YORK TIMES|SUNDAY TIMES|USA TODAY|#1|NUMBER ONE|"
    r"WINNER|AWARD|PRIZE|MAJOR (?:MOTION PICTURE|NETFLIX|TV)|NOW A |SOON TO BE|"
    r"OVER \d|MILLION COPIES|INSTANT|EDITION|BOOK CLUB"
)
# header of a reviews section: "Praise for I Am Zlatan", "Praise for the Temeraire series:"
_PRAISE_HEADER_RE = re.compile(
    rf"\b(?:advance\s+|critical\s+)?(?:praise|acclaim)\s+for\b[^.!?{_Q_OPEN}]{{0,120}}?(?::|(?=\s*[{_Q_OPEN}])|$)",
    flags=re.IGNORECASE,
)
_TOKEN_RE = re.compile(r"\S+")
# small words that can start the REAL sentence right after a banner in capitals
_CAPS_SENTENCE_STARTERS = {"A", "AN", "THE", "I", "IN", "ON", "AT", "AS", "IT", "HE", "SHE", "WE", "THEY"}


def _followed_by_lowercase(text: str, end: int) -> bool:
    """True if the text after `end` starts with a lowercase word (= we would cut a sentence)."""
    # commas and dashes are skipped: "Winner of the Pulitzer Prize--a powerful love story"
    # or "…Award, the #1 bestseller from…" are appositions, not a new sentence
    rest = text[end:].lstrip(" ,;-–—")
    return bool(rest) and rest[0].islower()


def _sub_leading_if_standalone(regex: re.Pattern, text: str) -> str:
    """Remove the banner at the start of the text ONLY if it forms a standalone segment.

    "#1 NEW YORK TIMES BESTSELLER - A pair of sisters…" -> removed;
    "The #1 New York Times bestselling debut novel that…" -> kept (the rest
    "debut novel that…" would no longer make sense).
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
    """Remove press quotes and commercial banners from a description.

    author_names / titles: lists taken from the dataset (authors.csv, titles),
    used to know where an attribution ends. Optional.
    """

    def __init__(self, author_names: Iterable[str] = (), titles: Iterable[str] = ()):
        self.names = {n for n in (norm_words(a) for a in author_names) if 2 <= len(n.split()) <= 5}
        self.titles = {t for t in (norm_words(x) for x in titles) if 1 <= len(t.split()) <= 12}

    # -- resolving the attribution ------------------------------------------
    @staticmethod
    def _publication_len(sub: str) -> int:
        """Length of the known press publication at the start of sub (0 otherwise)."""
        low = sub.lower()
        for pub in PUBLICATIONS:
            if low.startswith(pub) and (
                len(sub) == len(pub)
                or not sub[len(pub)].isalnum()
                # publication glued to the next word for lack of a space: "--TLSThe war raged…"
                or (sub[len(pub)].isupper() and sub[len(pub) - 1].isalpha()
                    and sub[:len(pub)].isupper() or (sub[len(pub)].isupper() and sub[len(pub) - 1].islower()))
            ):
                return len(pub)
        return 0

    def _source_end(self, sub: str) -> int:
        """Length of the "source" at the start of sub: publication, known author or likely name."""
        pub_len = self._publication_len(sub)
        if pub_len:
            return pub_len
        toks = list(_TOKEN_RE.finditer(sub[:160]))[:6]
        # known author name (longest first)
        if self.names:
            for n in range(min(5, len(toks)), 1, -1):
                if not _is_cap(toks[0].group()) or not all(
                        _is_cap(t.group()) or t.group().lower() in _NAME_PARTICLES for t in toks[:n]):
                    continue
                span = sub[toks[0].start():toks[n - 1].end()].rstrip(",;:")
                if norm_words(span) in self.names:
                    return toks[0].start() + len(span)
        # cautious fallback: "First [Initial] Last" (2 or 3 capitalised words), otherwise 1 word
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
        """Length of the work title(s) at the start of sub, after "author of"."""
        end = 0
        for _ in range(3):  # "author of A, B and C"
            lead = re.match(r"the\s+(?=[A-Z])", sub[end:])  # "author of the Pip & Flinx series"
            if lead:
                end += lead.end()
            rest = sub[end:]
            toks = list(_TOKEN_RE.finditer(rest[:200]))[:12]
            if not toks or not _is_cap(toks[0].group().lstrip("\"“'‘*_")):
                break
            # (1) known title from the dataset, as long as possible
            known, known_len = 0, 0
            if self.titles:
                for k in range(len(toks), 0, -1):
                    span = rest[toks[0].start():toks[k - 1].end()].rstrip(",;:.!?")
                    if norm_words(span) in self.titles:
                        known, known_len = toks[0].start() + len(span), k
                        break
            # (2) heuristic: capitalised words / small words, stopping before a
            #     capitalised word followed by a lowercase word (= start of the real sentence)
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
            # a capitalised function word at the end of the title actually opens the next sentence:
            # "author of Tiny Beautiful Things On the eve of…" -> "On" does not belong to the title
            while len(cap_tokens) > 1 and cap_tokens[-1].group().lower() in _SENTENCE_STARTERS:
                cap_tokens.pop()
            heuristic = cap_tokens[-1].end() if cap_tokens else 0
            # a known ONE-word title ("Sweet") does not beat a longer heuristic
            # match ("Sweet Evil"): many common words are also titles
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
        """End position of the attribution starting at `pos` (after the dash)."""
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
            # "Janet Maslin, The New York Times": name then publication
            m = re.match(r"\s*,\s*", sub[end:])
            if m:
                after = end + m.end()
                pub_len = self._publication_len(sub[after:])
                if pub_len:
                    end = after + pub_len
        return pos + end

    # -- cleaning steps ---------------------------------------------------
    def _strip_blurbs(self, text: str) -> str:
        pos = 0
        while True:
            m = _BLURB_START_RE.search(text, pos)
            if not m:
                return text
            end = self._attribution_end(text, m.end())
            if end == m.end():
                # no source after the dash (lowercase word): this is not a press
                # review but a quote INSIDE the story -> left untouched
                # e.g. "a Turkish inventor known as Hoja--\"master\"--a man who is his double"
                pos = m.end()
                continue
            text = text[:m.start()] + " " + text[end:].lstrip(" ,;.")
            pos = m.start()

    @staticmethod
    def _drop_sales_caps(match: re.Match) -> str:
        """Remove a banner in CAPITALS if it contains a sales pitch.

        If what follows starts with a lowercase letter ("NEW YORK TIMES BESTSELLER A superb
        love story"), the last word of the segment ("A") is actually the start of the
        real sentence: it is given back to the text. Otherwise everything is kept
        (we prefer leaving marketing to cutting a sentence).
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
        for _ in range(8):  # banners and quotes can follow each other
            before = text
            # 1) banners in CAPITALS first (otherwise the mixed-case regex would only
            #    remove their beginning: "THE INSTANT NEW YORK TIMES BESTSELLING | DEBUT…")
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


# Default instance, without lists (cautious heuristics)
_DEFAULT = MarketingStripper()


def strip_marketing(text: str, stripper: MarketingStripper | None = None) -> str:
    """Shortcut: remove the marketing text with `stripper` (or the default instance)."""
    return (stripper or _DEFAULT)(text)
