"""What a minor leaguer is worth in this league, on the same three-year scale as every other player.

    python -m breakout.prospect_value        # -> output/v3/prospect_values.csv (+ the "minors" block the site reads)

The league rule (Bort, 2026-09-28): besides the 5 hitters + 2 pitchers, a team may keep **2 minor leaguers who have not
played in the majors at all** (Fantrax's green flag). A minor-league slot that is not filled by a keeper is filled in the
minor-league draft. So a prospect is worth what he adds over the prospect that slot would otherwise hold.

Prospects are the hardest players to value because almost everything depends on WHEN and WHETHER they arrive, and a
projection system built on big-league seasons has nothing to say about that. So nothing here is assumed: every number
is read off what actually happened to minor leaguers who looked like him.

  cohorts   every minor leaguer 2021-2025 with 150+ PA (hitters) or 50+ IP (pitchers) at A through AAA who had not yet
            played in the majors, described by his highest level, how young he was for it and how well he played there
  outcome   what he was worth over the next three seasons in this league's points, the way a keeper is valued:
              year 1   points above replacement (3.67 a game for a hitter, 7.2 a start for a pitcher), floored at 0 because
                       a rookie who is not helping sits on the bench
              year 2-3 the same, BUT once he has debuted he has lost his minor-league slot and has to beat the league's
                       keeper line to be kept again, so only what he beats it by counts (exported on a grid of lines; the
                       page reads the line off its own board)
            discounted like every other player (0.8 a year, applied on the page)
  estimate  a weighted average over the most similar players in those cohorts (k nearest, distance-weighted), with k
            picked on held-out cohorts. Only cohorts old enough to have seen that season count toward it.
  pedigree  MLB Pipeline's rank is scouting information the stat line does not have, but there is no archive of past ranks
            to measure how much it adds. So a ranked prospect is moved halfway toward the value typical of prospects
            ranked near him (estimated on this year's list, from the same model), which smooths the stat line with the
            scouts without pretending to know their exact weight. A ranked draftee with no qualifying line (a teenager
            fresh out of the draft) gets the rank value alone.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import requests

from . import config as C
from .config import DATA
from .names import key

_S = requests.Session(); _S.headers.update({"User-Agent": "Mozilla/5.0"})
LEVELS = {11: "AAA", 12: "AA", 13: "A+", 14: "A"}
ORDER = {"AAA": 4, "AA": 3, "A+": 2, "A": 1}
LEVEL_AGE = {"AAA": 26.5, "AA": 24.5, "A+": 22.5, "A": 21.0}
FIRST = 2021
REPL_H, REPL_P = 3.67, 7.2
CUTS = [0, 50, 100, 150, 200, 250, 300, 350, 400, 500]


def _fetch(group: str, year: int, sport: int, refresh=False) -> list[dict]:
    p = DATA / "milb" / f"{group}_{LEVELS[sport]}_{year}.json"
    if p.exists() and not refresh:
        return json.loads(p.read_text())
    out, offset = [], 0
    while True:
        u = (f"https://statsapi.mlb.com/api/v1/stats?stats=season&group={group}&season={year}&sportId={sport}"
             f"&playerPool=all&limit=2000&offset={offset}&hydrate=person")
        st = _S.get(u, timeout=120).json()["stats"][0]; sp = st.get("splits", []); out += sp; offset += len(sp)
        if not sp or offset >= st.get("totalSplits", 0): break
    p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(out))
    return out


def _mlb(kind: str, year: int) -> list[dict]:
    f = DATA / "mlb" / (f"hitting_{year}.json" if kind == "h" else f"pitching_season_{year}.json")
    return json.loads(f.read_text()) if f.exists() else []


def _sp(year: int) -> list[dict]:
    f = DATA / "mlb" / f"pitching_split_sp_{year}.json"
    return json.loads(f.read_text()) if f.exists() else []


def _ip(x) -> float:
    s = str(x or "0"); w, _, f = s.partition("."); return float(w or 0) + (float(f or 0) / 3 if f else 0)


def _z(s: pd.Series) -> pd.Series:
    sd = s.std(ddof=0); return (s - s.mean()) / (sd if sd and sd > 0 else 1)


# ---------- minor-league seasons, one row per player-season at his highest real level
def milb_hitters(years) -> pd.DataFrame:
    rows = []
    for y in years:
        for sport, lvl in LEVELS.items():
            for s in _fetch("hitting", y, sport):
                p, st = s["player"], s["stat"]; pa = st.get("plateAppearances") or 0
                if pa < 1: continue
                ab = st.get("atBats") or 0; hr = st.get("homeRuns") or 0; d2 = st.get("doubles") or 0; d3 = st.get("triples") or 0
                rows.append(dict(mlbam_id=p["id"], name=p.get("fullName"), birth=p.get("birthDate"), season=y, level=lvl, PA=pa, AB=ab,
                                 H=st.get("hits") or 0, D2=d2, D3=d3, HR=hr, BB=(st.get("baseOnBalls") or 0) + (st.get("hitByPitch") or 0),
                                 K=st.get("strikeOuts") or 0, SB=st.get("stolenBases") or 0))
    d = pd.DataFrame(rows)
    d = d.groupby(["mlbam_id", "season", "level"], as_index=False).agg(name=("name", "first"), birth=("birth", "first"),
        **{c: (c, "sum") for c in ["PA", "AB", "H", "D2", "D3", "HR", "BB", "K", "SB"]})
    tot = d.groupby(["mlbam_id", "season"])["PA"].sum().rename("milb_PA_all")
    d = d[d["PA"] >= 100].copy()
    ab = d["AB"].clip(lower=1)
    d["OBP"] = (d["H"] + d["BB"]) / d["PA"]; d["ISO"] = (d["D2"] + 2 * d["D3"] + 3 * d["HR"]) / ab
    d["BBK"] = (d["BB"] - d["K"]) / d["PA"]; d["SBr"] = d["SB"] / d["PA"]
    # a points-league bat within its level and season: power, on-base, discipline, speed (as prospects.py)
    d["bat_z"] = d.groupby(["season", "level"], group_keys=False).apply(
        lambda g: (_z(g["ISO"]) + _z(g["OBP"]) + _z(g["BBK"]) + 0.5 * _z(g["SBr"])) / 3.5)
    d["lvl"] = d["level"].map(ORDER)
    top = d.sort_values(["mlbam_id", "season", "lvl", "PA"], ascending=[True, True, False, False]).drop_duplicates(["mlbam_id", "season"])
    top = top.merge(tot, on=["mlbam_id", "season"], how="left")
    top["age"] = [(pd.Timestamp(f"{s}-06-30") - pd.Timestamp(b)).days / 365.25 if isinstance(b, str) else np.nan for s, b in zip(top["season"], top["birth"])]
    top["young"] = top["level"].map(LEVEL_AGE) - top["age"]
    return top


def milb_pitchers(years) -> pd.DataFrame:
    rows = []
    for y in years:
        for sport, lvl in LEVELS.items():
            for s in _fetch("pitching", y, sport):
                p, st = s["player"], s["stat"]; ip = _ip(st.get("inningsPitched")); bf = st.get("battersFaced") or 0
                if bf < 1: continue
                rows.append(dict(mlbam_id=p["id"], name=p.get("fullName"), birth=p.get("birthDate"), season=y, level=lvl, IP=ip, BF=bf,
                                 GS=st.get("gamesStarted") or 0, K=st.get("strikeOuts") or 0, BB=(st.get("baseOnBalls") or 0) + (st.get("hitByPitch") or 0),
                                 HR=st.get("homeRuns") or 0, ER=st.get("earnedRuns") or 0))
    d = pd.DataFrame(rows)
    d = d.groupby(["mlbam_id", "season", "level"], as_index=False).agg(name=("name", "first"), birth=("birth", "first"),
        **{c: (c, "sum") for c in ["IP", "BF", "GS", "K", "BB", "HR", "ER"]})
    tot = d.groupby(["mlbam_id", "season"])["IP"].sum().rename("milb_IP_all")
    d = d[d["IP"] >= 30].copy()
    d["KBB"] = (d["K"] - d["BB"]) / d["BF"]; d["HRr"] = d["HR"] / d["BF"]; d["ERA"] = 9 * d["ER"] / d["IP"].clip(lower=1)
    d["arm_z"] = d.groupby(["season", "level"], group_keys=False).apply(lambda g: (2 * _z(g["KBB"]) - _z(g["HRr"]) - 0.5 * _z(g["ERA"])) / 3.5)
    d["lvl"] = d["level"].map(ORDER)
    top = d.sort_values(["mlbam_id", "season", "lvl", "IP"], ascending=[True, True, False, False]).drop_duplicates(["mlbam_id", "season"])
    top = top.merge(tot, on=["mlbam_id", "season"], how="left")
    top["gs_share"] = top["GS"] / top.groupby(["mlbam_id", "season"])["GS"].transform("sum").clip(lower=1)
    top["age"] = [(pd.Timestamp(f"{s}-06-30") - pd.Timestamp(b)).days / 365.25 if isinstance(b, str) else np.nan for s, b in zip(top["season"], top["birth"])]
    top["young"] = top["level"].map(LEVEL_AGE) - top["age"]
    return top


# ---------- big-league outcomes in this league's points
def mlb_hitting(years) -> pd.DataFrame:
    rows = []
    for y in years:
        for s in _mlb("h", y):
            st = s["stat"]; h = st.get("hits") or 0; d2 = st.get("doubles") or 0; d3 = st.get("triples") or 0; hr = st.get("homeRuns") or 0
            pts = 2 * (h - d2 - d3 - hr) + 3 * d2 + 4 * d3 + 5 * hr + (st.get("runs") or 0) + (st.get("rbi") or 0) + (st.get("baseOnBalls") or 0) \
                + (st.get("hitByPitch") or 0) + 3 * (st.get("stolenBases") or 0)
            rows.append(dict(mlbam_id=s["player"]["id"], season=y, PA=st.get("plateAppearances") or 0, G=st.get("gamesPlayed") or 0, pts=pts))
    d = pd.DataFrame(rows).groupby(["mlbam_id", "season"], as_index=False).sum()
    d["par"] = d["pts"] - REPL_H * d["G"]
    return d


def mlb_pitching(years) -> pd.DataFrame:
    app, rows = [], []
    for y in years:
        for s in _mlb("p", y): app.append(dict(mlbam_id=s["player"]["id"], season=y, app=s["stat"].get("gamesPlayed") or 0, QS=s["stat"].get("qualityStarts") or 0))
        for s in _sp(y):
            st = s["stat"]; gs = st.get("gamesStarted") or 0
            pts = _ip(st.get("inningsPitched")) + (st.get("strikeOuts") or 0) - (st.get("earnedRuns") or 0) + 10 * (st.get("completeGames") or 0) \
                + 8 * (st.get("shutouts") or 0) - 3 * (st.get("blownSaves") or 0)
            rows.append(dict(mlbam_id=s["player"]["id"], season=y, GS=gs, pts=pts))
    a = pd.DataFrame(app).groupby(["mlbam_id", "season"], as_index=False).sum() if app else pd.DataFrame(columns=["mlbam_id", "season", "app"])
    d = pd.DataFrame(rows).groupby(["mlbam_id", "season"], as_index=False).sum() if rows else pd.DataFrame(columns=["mlbam_id", "season", "GS", "pts"])
    d = a.merge(d, on=["mlbam_id", "season"], how="left").fillna({"GS": 0, "pts": 0})
    d["pts"] = d["pts"] + 4 * d["QS"]          # quality starts only come in the season feed (every QS is a start)
    d["par"] = d["pts"] - REPL_P * d["GS"]
    return d


# ---------- the model
FEAT = ["lvl", "young", "z"]
SCALE = np.array([1.0, 1.0 / 1.5, 1.0 / 0.8])   # one level ~ 1.5 years of age-for-level ~ 0.8 of a within-level z


def _targets(coh: pd.DataFrame, mlb: pd.DataFrame, first: dict, last: int) -> pd.DataFrame:
    """Realized value in years s+1..s+3, on a grid of keeper lines for the years after a debut."""
    par = mlb.set_index(["mlbam_id", "season"])["par"].to_dict()
    out = coh.copy()
    for k in (1, 2, 3):
        yr = out["season"] + k; ok = yr <= last
        p = np.array([par.get((i, y), 0.0) for i, y in zip(out["mlbam_id"], yr)])
        deb = np.array([first.get(i, 9999) for i in out["mlbam_id"]])
        out[f"ok{k}"] = ok
        if k == 1:
            out["y1"] = np.where(ok, np.maximum(0, p), np.nan)
            out["deb1"] = np.where(ok, (deb <= yr).astype(float), np.nan)
        else:
            before = deb <= (yr - 1)          # debuted before this season began: no minor-league slot any more
            for c in CUTS:
                out[f"y{k}_{c}"] = np.where(ok, np.maximum(0, p - np.where(before, c, 0)), np.nan)
    return out


def _knn(train: pd.DataFrame, test: pd.DataFrame, cols: list[str], k: int) -> tuple[np.ndarray, list]:
    X = train[FEAT].to_numpy() * SCALE; T = test[FEAT].to_numpy() * SCALE
    Y = train[cols].to_numpy(); res = np.zeros((len(T), len(cols))); nbrs = []
    for i, t in enumerate(T):
        d = np.sqrt(((X - t) ** 2).sum(1)); idx = np.argsort(d)[:k]; w = 1.0 / (0.15 + d[idx])
        res[i] = (w[:, None] * Y[idx]).sum(0) / w.sum(); nbrs.append(idx[:3])
    return res, nbrs


def _pick_k(coh: pd.DataFrame, col: str, okcol: str) -> int:
    """k by held-out cohort: fit on earlier seasons, score the latest one that has the outcome."""
    d = coh[coh[okcol]]
    seasons = sorted(d["season"].unique())
    if len(seasons) < 2: return 60
    best, bk = None, 60
    for k in (20, 40, 60, 100, 160):
        err = []
        for s in seasons[1:]:
            tr, te = d[d["season"] < s], d[d["season"] == s]
            if len(tr) < k: continue
            pred, _ = _knn(tr, te, [col], k); err.append(((pred[:, 0] - te[col].to_numpy()) ** 2).mean())
        if err and (best is None or np.mean(err) < best): best, bk = np.mean(err), k
    return bk


def value(kind: str, cur_season: int) -> tuple[pd.DataFrame, dict]:
    yrs = list(range(FIRST, cur_season + 1))
    if kind == "h":
        mi = milb_hitters(yrs); mi = mi[mi["PA"] >= 150].rename(columns={"bat_z": "z"}); mlb = mlb_hitting(yrs)
        played = mlb[mlb["PA"] > 0]
    else:
        mi = milb_pitchers(yrs); mi = mi[(mi["IP"] >= 50) & (mi["gs_share"] >= 0.5)].rename(columns={"arm_z": "z"}); mlb = mlb_pitching(yrs)
        played = mlb[mlb["app"] > 0]
    mi = mi.dropna(subset=["age"]); mi = mi[mi["age"] <= 26]
    # eligible = had not played in the majors by the end of that minor-league season
    first = played.groupby("mlbam_id")["season"].min().to_dict()
    mi["eligible"] = [first.get(i, 9999) > s for i, s in zip(mi["mlbam_id"], mi["season"])]
    coh = _targets(mi[mi["eligible"] & (mi["season"] < cur_season)], mlb, first, cur_season)
    now = mi[mi["season"] == cur_season].copy()
    ks = {}; stats = {}
    for tag, okc in (("y1", "ok1"), ("y2_0", "ok2"), ("y3_0", "ok3")):
        ks[tag] = _pick_k(coh, tag, okc)
    out = now[["mlbam_id", "name", "age", "level", "lvl", "young", "z"] + (["PA"] if kind == "h" else ["IP"])].copy()
    comps = {}
    for yk, okc in (("y1", "ok1"), ("y2", "ok2"), ("y3", "ok3")):
        tr = coh[coh[okc]].reset_index(drop=True)
        cols = ["y1", "deb1"] if yk == "y1" else [f"{yk}_{c}" for c in CUTS]
        k = ks["y1" if yk == "y1" else f"{yk}_0"]
        pred, nb = _knn(tr, out, cols, k)
        for j, c in enumerate(cols): out[c] = pred[:, j]
        stats[yk] = dict(k=k, n=int(len(tr)), seasons=sorted(int(s) for s in tr["season"].unique()))
        if yk == "y1":
            for i, idx in zip(out["mlbam_id"], nb):
                comps[i] = [dict(name=tr.loc[j, "name"], season=int(tr.loc[j, "season"]), level=tr.loc[j, "level"], age=round(float(tr.loc[j, "age"]), 1),
                                 y1=round(float(tr.loc[j, "y1"]), 0)) for j in idx]
    out["comps"] = out["mlbam_id"].map(comps)
    return out, dict(stats=stats, cohort=int(len(coh)), k=ks)


def _pipeline() -> pd.DataFrame:
    p = DATA / "prospects" / "pipeline_current.json"
    d = json.loads(p.read_text()) if p.exists() else []
    return pd.DataFrame([dict(mlbam_id=x["playerId"], pl_name=x["name"], pipeline_rank=x["rank"], pl_pos=x.get("position"), pl_age=x.get("age"),
                              pl_team=(x.get("team") or "").upper()) for x in d])


def _ever_played() -> set:
    ids = set()
    for y in range(FIRST, C.CURRENT_SEASON + 1):
        ids |= {s["player"]["id"] for s in _mlb("h", y) if (s["stat"].get("plateAppearances") or 0) > 0}
        ids |= {s["player"]["id"] for s in _mlb("p", y) if (s["stat"].get("gamesPlayed") or 0) > 0}
    return ids


def build(cur_season: int | None = None) -> tuple[pd.DataFrame, dict]:
    cur = cur_season or C.CURRENT_SEASON
    h, mh = value("h", cur); p, mp = value("p", cur)
    h["kind"] = "h"; p["kind"] = "p"
    allp = pd.concat([h, p], ignore_index=True)
    pl = _pipeline(); played = _ever_played()
    allp = allp.merge(pl[["mlbam_id", "pipeline_rank", "pl_pos"]], on="mlbam_id", how="left")
    # ranked prospects with no qualifying line this year (a draftee in rookie ball, a hitter hurt most of the season)
    miss = pl[~pl["mlbam_id"].isin(allp["mlbam_id"])].copy()
    if len(miss):
        miss["kind"] = np.where(miss["pl_pos"].isin(["RHP", "LHP", "P"]), "p", "h")
        miss = miss.rename(columns={"pl_name": "name", "pl_age": "age"})[["mlbam_id", "name", "age", "kind", "pipeline_rank", "pl_pos"]]
        allp = pd.concat([allp, miss], ignore_index=True)
    allp["eligible"] = ~allp["mlbam_id"].isin(played)
    ycols = ["y1"] + [f"y2_{c}" for c in CUTS] + [f"y3_{c}" for c in CUTS]
    # pedigree: halfway to the value typical of prospects ranked near him (a rolling window over this year's ranked list)
    allp["stat_only"] = allp["y1"].notna()
    for kind in ("h", "p"):
        m = (allp["kind"] == kind) & allp["pipeline_rank"].notna()
        rk = allp.loc[m & allp["stat_only"]].sort_values("pipeline_rank")
        if len(rk) < 5: continue
        for c in ycols:
            s = rk.set_index("pipeline_rank")[c]
            def at(r, s=s):
                w = np.exp(-((s.index.to_numpy() - r) / 12.0) ** 2); return float((w * s.to_numpy()).sum() / w.sum())
            typ = allp.loc[m, "pipeline_rank"].map(at)
            has = allp.loc[m, c].notna()
            allp.loc[m, c] = np.where(has, 0.5 * allp.loc[m, c].fillna(0) + 0.5 * typ, typ)
    allp[ycols] = allp[ycols].fillna(0).round(2)
    allp["deb1"] = allp["deb1"].round(3)
    allp["headline"] = allp["y1"] + 0.8 * allp["y2_100"] + 0.64 * allp["y3_100"]   # near the keeper lines the page uses
    allp = allp.sort_values("headline", ascending=False)
    meta = dict(hitters=mh, pitchers=mp, cuts=CUTS, first=FIRST, season=cur)
    return allp, meta


def to_block(df: pd.DataFrame, meta: dict, keep_names: set | None = None, top: int = 150) -> dict:
    """The page's copy: every rostered prospect plus the best `top` eligible ones in baseball (the minor-league pool,
    which sets what a minor-league draft pick is worth)."""
    keep_names = keep_names or set()
    d = df[df["eligible"]].head(top)
    d = pd.concat([d, df[df["name"].map(key).isin(keep_names)]]).drop_duplicates("mlbam_id")
    rows = []
    for r in d.itertuples():
        rows.append(dict(mlbam_id=int(r.mlbam_id), name=r.name, kind=r.kind, age=None if pd.isna(r.age) else round(float(r.age), 1),
                         level=None if pd.isna(getattr(r, "level", np.nan)) else r.level, rank=None if pd.isna(r.pipeline_rank) else int(r.pipeline_rank),
                         eligible=bool(r.eligible), stat=bool(r.stat_only), z=None if pd.isna(getattr(r, "z", np.nan)) else round(float(r.z), 2),
                         deb1=None if pd.isna(r.deb1) else float(r.deb1), y1=float(r.y1),
                         y2=[float(getattr(r, f"y2_{c}")) for c in CUTS], y3=[float(getattr(r, f"y3_{c}")) for c in CUTS],
                         comps=r.comps if isinstance(r.comps, list) else []))
    return dict(cuts=CUTS, rows=rows, meta=meta)


def augment(data: dict) -> dict:
    """Add the "minors" block to explorer_data from the last model run (output/v3/prospect_values.csv). The run itself
    needs six years of minor-league downloads, so it happens in the weekly full rebuild, not on every refresh."""
    f = C.OUT / "v3" / "prospect_values.csv"
    if not f.exists(): return data
    df = pd.read_csv(f)
    cf = C.OUT / "v3" / "prospect_comps.json"; comps = json.loads(cf.read_text()) if cf.exists() else {}
    df["comps"] = df["mlbam_id"].map(lambda i: comps.get(str(int(i)), []))
    mf = C.OUT / "v3" / "prospect_values_meta.json"; meta = json.loads(mf.read_text()) if mf.exists() else {}
    names = set()
    ro = C.DATA / "fantrax" / f"rosters_{C.CURRENT_SEASON}.csv"
    if ro.exists(): names = set(pd.read_csv(ro)["player"].dropna().map(key))
    data["minors"] = to_block(df, {k: meta.get(k) for k in ("season", "first", "cuts")} | {
        "k": {"h": (meta.get("hitters") or {}).get("k"), "p": (meta.get("pitchers") or {}).get("k")},
        "cohort": {"h": (meta.get("hitters") or {}).get("cohort"), "p": (meta.get("pitchers") or {}).get("cohort")}}, names)
    return data


def main(argv=None):
    df, meta = build()
    out = C.OUT / "v3"; out.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["comps"]).to_csv(out / "prospect_values.csv", index=False)
    (out / "prospect_values_meta.json").write_text(json.dumps(meta, indent=1))
    comps = {int(r.mlbam_id): r.comps for r in df.itertuples() if isinstance(r.comps, list)}
    (out / "prospect_comps.json").write_text(json.dumps(comps))
    print(json.dumps(meta, indent=1))
    show = df[df["eligible"]].head(40)
    print(show[["name", "kind", "age", "level", "pipeline_rank", "z", "deb1", "y1", "y2_100", "y3_100", "headline"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
