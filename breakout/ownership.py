"""Who owns whom, as of the last Fantrax sync, stamped on top of the projection tables.

pipeline3 writes an `owner` column when the model is rebuilt (rarely). fantrax_api.sync() runs every morning and writes
data/fantrax/rosters_<season>.csv. Anything that shows ownership (the site, the artifact editions, the streamer tables)
must take it from the sync, not from the projection tables, so a player dropped yesterday is a free agent today
without rebuilding the model. Everything here is read-only.

    ownership.owners()            -> Owners (lookup by name, optionally by MLB club for duplicate names)
    ownership.apply(data)         -> explorer_data dict with fresh owner / likely_kept fields
    ownership.stamp(df, name_col) -> DataFrame with a fresh `owner` column
"""
from __future__ import annotations
import datetime as dt, json
import pandas as pd
from . import config as C
from .names import key

FREE = ("FA", "W (Sun)", "W (Mon)")


class Owners:
    def __init__(self, ro: pd.DataFrame, synced: str):
        ro = ro.copy(); ro["nkey"] = ro["player"].apply(key)
        self.dup = set(ro.loc[ro["nkey"].duplicated(keep=False), "nkey"])
        self.uniq = {r.nkey: r.owner for r in ro[~ro["nkey"].isin(self.dup)].itertuples()}
        self.by_club = {(r.nkey, str(r.mlb).upper()): r.owner for r in ro.itertuples()}
        self.synced = synced; self.n = len(ro)

    def get(self, name: str, club: str | None = None, default: str = "FA", ambiguous: bool = False) -> str:
        """Owner of `name`. `club` (MLB abbreviation) breaks ties when Fantrax rosters two players with the same name, and
        is required to match when the caller knows the name is shared by two MLB players (`ambiguous`: two Max Muncys,
        only one rostered) so the unrostered namesake stays a free agent."""
        k = key(name); c = str(club).upper() if club else None
        if c and (k, c) in self.by_club: return self.by_club[(k, c)]
        if ambiguous and c: return default
        if k in self.uniq: return self.uniq[k]
        vals = {v for (kk, _), v in self.by_club.items() if kk == k}
        return vals.pop() if len(vals) == 1 else default


def owners(season: int | None = None) -> Owners:
    season = season or C.CURRENT_SEASON; d = C.DATA / "fantrax"
    p = d / f"rosters_{season}.csv"
    if not p.exists(): p = d / f"hitters_{season}.csv"          # same content, older layout
    ro = pd.read_csv(p)[["player", "mlb", "owner"]]
    meta = d / "sync_meta.json"   # written by fantrax_api.sync(); fall back to the file's modification time
    synced = json.loads(meta.read_text()).get("synced_utc") if meta.exists() else None
    synced = synced or dt.datetime.fromtimestamp(p.stat().st_mtime, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return Owners(ro, synced)


def stamp(df: pd.DataFrame, name_col: str = "name", club_col: str | None = None, ow: Owners | None = None) -> pd.DataFrame:
    """Replace (or add) df['owner'] from the sync. Rows whose name is unknown to Fantrax are free agents."""
    ow = ow or owners(); df = df.copy()
    clubs = df[club_col] if club_col and club_col in df.columns else pd.Series([None] * len(df), index=df.index)
    keys = df[name_col].apply(key); shared = set(keys[keys.duplicated(keep=False)])
    df["owner"] = [ow.get(n, c, ambiguous=k in shared) for n, c, k in zip(df[name_col], clubs, keys)]
    return df


def _columnar_names(block: dict, name_col: str = "name", club_col: str | None = None, season: int | None = None) -> dict:
    """mlbam_id -> (name, club) from a columnar block (latest season wins)."""
    cols = block.get("cols", []); rows = block.get("rows", []); out = {}
    if "mlbam_id" not in cols or name_col not in cols: return out
    i_id, i_n = cols.index("mlbam_id"), cols.index(name_col); i_c = cols.index(club_col) if club_col in cols else None; i_s = cols.index("season") if "season" in cols else None
    for r in sorted(rows, key=lambda r: r[i_s] if i_s is not None else 0):
        out[r[i_id]] = (r[i_n], r[i_c] if i_c is not None else None)
    return out


def _likely_kept(block: dict, n_keep: int, pitcher_slot: set[str] | None = None, names: dict | None = None, home: str | None = None) -> None:
    """Recompute likely_kept in a columnar projection block: each owner's top n_keep by KSV (ties by proj_pts).
    A two-way player the home team keeps in a pitcher slot (Ohtani) is kept, but not off the hitter board."""
    cols = block["cols"]
    if not {"owner", "KSV", "likely_kept"} <= set(cols): return
    i_o, i_k, i_l = cols.index("owner"), cols.index("KSV"), cols.index("likely_kept"); i_p = cols.index("proj_pts") if "proj_pts" in cols else i_k; i_id = cols.index("mlbam_id")
    def pslot(r): return bool(pitcher_slot) and r[i_o] == home and key((names or {}).get(r[i_id], ("", None))[0]) in pitcher_slot
    by = {}
    for r in block["rows"]:
        o = r[i_o]
        if o in FREE or not o or pslot(r): continue
        by.setdefault(o, []).append(r)
    kept = set()
    for o, rs in by.items():
        rs.sort(key=lambda r: ((r[i_k] if r[i_k] is not None else -1e9), (r[i_p] if r[i_p] is not None else -1e9)), reverse=True)
        kept.update(id(r) for r in rs[:n_keep])
    for r in block["rows"]:
        r[i_l] = bool(id(r) in kept or pslot(r))


def apply(data: dict, ow: Owners | None = None) -> dict:
    """Overwrite ownership in explorer_data (projections, pitcher_proj, prospects, pool, pitcher_pool) from the sync."""
    ow = ow or owners(); meta = data.get("meta", {})
    hk = int(meta.get("hitter_keepers", 5)); pk = int(meta.get("pitcher_keepers", 2))
    v3 = C.OUT / "v3"; hn = {}; pn = {}
    hp = v3 / f"hitter_projections_{C.CURRENT_SEASON + 1}.csv"; pp = v3 / f"pitcher_projections_{C.CURRENT_SEASON + 1}.csv"
    if hp.exists():
        h = pd.read_csv(hp, usecols=lambda c: c in ("mlbam_id", "name", "team_abbr")); hn = {int(r.mlbam_id): (r.name, getattr(r, "team_abbr", None)) for r in h.itertuples()}
    if pp.exists():
        p = pd.read_csv(pp, usecols=lambda c: c in ("mlbam_id", "name")); pn = {int(r.mlbam_id): (r.name, None) for r in p.itertuples()}
    hn = {**_columnar_names(data.get("seasons", {}), "name", "team_abbr"), **hn}
    pn = {**_columnar_names(data.get("pitchers", {}), "name"), **pn}
    # names shared by two different MLB players (two Max Muncys): the club has to match for either to be "owned"
    cnt = {}
    for nm, _ in list(hn.values()) + list(pn.values()): cnt[key(nm)] = cnt.get(key(nm), 0) + 1
    shared = {k for k, n in cnt.items() if n > 1}
    changed = 0
    for blk, names in (("projections", hn), ("pitcher_proj", pn)):
        b = data.get(blk)
        if not b or "owner" not in b.get("cols", []): continue
        i_o, i_id = b["cols"].index("owner"), b["cols"].index("mlbam_id")
        for r in b["rows"]:
            nm, club = names.get(r[i_id], (None, None))
            if nm is None: continue
            new = ow.get(nm, club, ambiguous=key(nm) in shared)
            if new != r[i_o]: changed += 1
            r[i_o] = new
    pitcher_slot = {key(k["player"]) for k in meta.get("real_keepers", []) if k.get("slot") == "pitcher"}
    if data.get("projections"): _likely_kept(data["projections"], hk, pitcher_slot, hn, home=meta.get("my_abbrev"))
    if data.get("pitcher_proj"): _likely_kept(data["pitcher_proj"], pk)
    for blk in ("prospects", "pool", "pitcher_pool"):
        for r in data.get(blk, []) or []:
            if "name" in r:
                new = ow.get(r["name"])
                if new != r.get("owner"): changed += 1
                r["owner"] = new
    meta["owners_synced"] = ow.synced; meta["owners_changed"] = changed; data["meta"] = meta
    return data


def stamp_streamers(s: dict, ow: Owners | None = None) -> dict:
    """Fresh owners on the streamer payload (hitter_days, hitter_week, pitcher_starts rows carry a name and an owner)."""
    ow = ow or owners()
    for blk in ("hitter_days", "hitter_week", "pitcher_starts"):
        for r in s.get(blk, []) or []:
            if r.get("name"): r["owner"] = ow.get(r["name"])
    s.setdefault("meta", {})["owners_synced"] = ow.synced
    return s


def main(argv=None):
    import sys
    ow = owners(); print(f"{ow.n} rostered players, synced {ow.synced}, {len(ow.dup)} duplicate names")
    data = json.load(open(C.OUT / "v3" / "explorer_data.json"))
    apply(data, ow); print("owner fields changed vs the projection tables:", data["meta"]["owners_changed"])
    for nm in (argv or [])[:10]: print(nm, "->", ow.get(nm))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main(sys.argv[1:]))
