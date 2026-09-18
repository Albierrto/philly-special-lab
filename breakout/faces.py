"""Player headshots: cached in the repo, embedded for the men on the twelve rosters.

The page asks MLB's image CDN for a face by player id, which is free and always current. That works on the site but
not inside a Claude artifact, where outside hosts are blocked, and not at all if you open the page on a plane. So the
players you actually look at every day, the ones somebody in the league rosters, travel with the build as small inline
images; everyone else gets the CDN URL and falls back to initials when it is unreachable or the man has no photo.

The cache lives in data/faces/ so a build re-downloads only the players who changed hands since the last one, and
ids MLB has no photo for are remembered in _missing.json rather than asked for again every morning.
"""
from __future__ import annotations
import base64, json
from pathlib import Path

import requests

from .config import DATA

SIZE = 72
URL = ("https://img.mlbstatic.com/mlb-photos/image/upload/"
       "w_{s},h_{s},c_fill,g_face,q_auto:good,f_jpg/v1/people/{i}/headshot/67/current")
FREE = ("FA", "W (Sun)", "W (Mon)", "", None)
DIR = DATA / "faces"
MISSING = DIR / "_missing.json"


def owned_ids(data: dict) -> list[int]:
    """Every player id that somebody in the league rosters, read out of the assembled tables themselves.

    Ownership is stamped on several blocks (projections, pitchers, the streamer days) and a man can appear in more
    than one, so this walks whatever is there rather than naming them, and stays correct when a block is added.
    """
    out: set[int] = set()

    def eat(block):
        if isinstance(block, dict) and "cols" in block and "rows" in block:
            cols = block["cols"]
            if "mlbam_id" not in cols or "owner" not in cols:
                return
            i, o = cols.index("mlbam_id"), cols.index("owner")
            for r in block["rows"]:
                if r[o] not in FREE and r[i] is not None:
                    out.add(int(r[i]))
        elif isinstance(block, list) and block and isinstance(block[0], dict):
            if "mlbam_id" in block[0] and "owner" in block[0]:
                for r in block:
                    if r.get("owner") not in FREE and r.get("mlbam_id") is not None:
                        out.add(int(r["mlbam_id"]))
        elif isinstance(block, dict):
            for v in block.values():
                eat(v)

    eat(data)
    return sorted(out)


def _cached(i: int) -> Path:
    return DIR / f"{i}.jpg"


def fetch(ids, timeout: float = 8.0) -> int:
    """Download any headshot not already cached. Returns how many were added."""
    DIR.mkdir(parents=True, exist_ok=True)
    try:
        missing = set(json.loads(MISSING.read_text()))
    except Exception:
        missing = set()
    got = 0
    for i in ids:
        if _cached(i).exists() or i in missing:
            continue
        try:
            r = requests.get(URL.format(s=SIZE, i=i), timeout=timeout)
        except Exception:
            continue                                   # the build never fails over a picture
        if r.status_code == 200 and r.content[:2] == b"\xff\xd8":
            _cached(i).write_bytes(r.content); got += 1
        elif r.status_code == 404:
            missing.add(int(i))
    try:
        MISSING.write_text(json.dumps(sorted(missing)))
    except Exception:
        pass
    return got


def embed(ids) -> dict[str, str]:
    """{id: data URI} for every id with a cached headshot. Keys are strings because this ends up in JSON."""
    out = {}
    for i in ids:
        p = _cached(i)
        if p.exists():
            b = p.read_bytes()
            if len(b) < 400_000:
                out[str(i)] = "data:image/jpeg;base64," + base64.b64encode(b).decode()
    return out


def build(data: dict, download: bool = True) -> dict[str, str]:
    ids = owned_ids(data)
    if download:
        fetch(ids)
    return embed(ids)
