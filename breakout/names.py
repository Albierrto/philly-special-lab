"""Name normalisation so Savant / MLB API / FantasyPros / Fantrax rows can be joined."""
from __future__ import annotations
import re, unicodedata

_SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv)\b\.?", re.I)


def norm(name: str) -> str:
    if not isinstance(name, str):
        return ""
    n = name.strip()
    if "," in n and not n.lower().endswith(", jr.") and n.count(",") == 1:   # "Last, First" (Savant)
        last, first = [x.strip() for x in n.split(",", 1)]
        n = f"{first} {last}"
    n = unicodedata.normalize("NFKD", n).encode("ascii", "ignore").decode()
    n = n.lower()
    n = re.sub(r"\(.*?\)", " ", n)              # "(Batter)"
    n = _SUFFIX.sub(" ", n)
    n = re.sub(r"[^a-z0-9 ]", "", n)           # drop punctuation ( T.J. -> tj, O'Neill -> oneill )
    n = re.sub(r"\s+", " ", n).strip()
    return n


# Known cross-source aliases (normalised form -> canonical normalised form)
ALIASES = {
    "jazz chisholm": "jazz chisholm",
    "michael harris": "michael harris",
    "luis garcia": "luis garcia",
    "julio rodriguez": "julio rodriguez",
    "jose ramirez": "jose ramirez",
    "eloy jimenez": "eloy jimenez",
    "cj abrams": "cj abrams",
    "jj bleday": "jj bleday",
    "tj friedl": "tj friedl",
    "jp crawford": "jp crawford",
    "jt realmuto": "jt realmuto",
    "ha seong kim": "haseong kim",
    "jung hoo lee": "jung hoo lee",
    "hyeseong kim": "hyeseong kim",
    "hye seong kim": "hyeseong kim",
    "shohei ohtani": "shohei ohtani",
}


def key(name: str) -> str:
    n = norm(name)
    return ALIASES.get(n, n)
