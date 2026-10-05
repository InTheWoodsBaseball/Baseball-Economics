"""
retrosheet_pull.py
------------------------------------------------------------------------------
Pull per-game ATTENDANCE + RECORDS and derive GAMES-BACK (division + playoff cut)
for every MLB team, every regular-season game, over a set of seasons.

SOURCE: Retrosheet game logs  (https://www.retrosheet.org/gamelogs/)
  - Free bulk downloads -- one small zip per season, no scraping of
    Baseball-Reference, no rate-limit jail.
  - Attendance, scores, date, park, day/night come straight from the logs.
  - Standings / games-back are COMPUTED from the same results, so you don't
    need a second source for competitiveness-by-standings.

NOT produced here: daily playoff ODDS. Those are not published anywhere as a
clean historical daily series; they must be simulated (Monte Carlo from daily
standings + a team-strength estimate). See the note at the bottom.

Requires:  pip install pandas requests numpy
Note: this was NOT executed against the live site in the environment it was
written in (no network there). Before trusting a full run, do the two sanity
checks in validate() at the bottom.

Retrosheet data use: free for research/public use with attribution to
Retrosheet (www.retrosheet.org). Keep that credit in anything you publish.
------------------------------------------------------------------------------
"""

import io
import os
import zipfile

import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEASONS = list(range(2016, 2026))          # last 10 completed seasons (2016-2025)
RETROSHEET_BASE = "https://www.retrosheet.org/gamelogs"
CACHE_DIR = "retrosheet_cache"
OUT_CSV = "mlb_team_game_panel.csv"
UA = {"User-Agent": "mlb-strategy-research/1.0 (personal research)"}

# Wild-card spots per league by season. The 3-WC format began in 2022.
# 2020 was a 60-game, 16-team, NO-FANS season -- exclude it from any attendance
# analysis; GB here is meaningless for 2020.
def wc_spots(season: int) -> int:
    if season == 2020:
        return 8  # placeholder; do not trust 2020 GB
    return 3 if season >= 2022 else 2

# Retrosheet team codes -> (league, division). Stable across 2016-2025.
DIVISIONS = {
    # AL East
    "BAL": ("AL", "E"), "BOS": ("AL", "E"), "NYA": ("AL", "E"),
    "TBA": ("AL", "E"), "TOR": ("AL", "E"),
    # AL Central
    "CHA": ("AL", "C"), "CLE": ("AL", "C"), "DET": ("AL", "C"),
    "KCA": ("AL", "C"), "MIN": ("AL", "C"),
    # AL West
    "HOU": ("AL", "W"), "ANA": ("AL", "W"), "OAK": ("AL", "W"),
    "SEA": ("AL", "W"), "TEX": ("AL", "W"),
    # NL East
    "ATL": ("NL", "E"), "MIA": ("NL", "E"), "NYN": ("NL", "E"),
    "PHI": ("NL", "E"), "WAS": ("NL", "E"),
    # NL Central
    "CHN": ("NL", "C"), "CIN": ("NL", "C"), "MIL": ("NL", "C"),
    "PIT": ("NL", "C"), "SLN": ("NL", "C"),
    # NL West
    "ARI": ("NL", "W"), "COL": ("NL", "W"), "LAN": ("NL", "W"),
    "SDN": ("NL", "W"), "SFN": ("NL", "W"),
}

# Game-log columns we need (0-based indices).
# Full field spec: https://www.retrosheet.org/gamelogs/glfields.txt
COL = {
    "date": 0, "game_num": 1, "vis_team": 3, "home_team": 6,
    "vis_score": 9, "home_score": 10, "day_night": 12,
    "park": 16, "attendance": 17,
}


# ---------------------------------------------------------------------------
# Download + parse
# ---------------------------------------------------------------------------
def download_gamelog(season: int) -> str:
    """Download + cache one season's regular-season game log (from the yearly zip)."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    local = os.path.join(CACHE_DIR, f"GL{season}.TXT")
    if os.path.exists(local):
        return local
    url = f"{RETROSHEET_BASE}/gl{season}.zip"
    r = requests.get(url, timeout=60, headers=UA)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        name = next(n for n in z.namelist() if n.upper().endswith(".TXT"))
        with z.open(name) as f, open(local, "wb") as out:
            out.write(f.read())
    return local


def parse_gamelog(season: int) -> pd.DataFrame:
    path = download_gamelog(season)
    idx_to_name = {v: k for k, v in COL.items()}
    use = sorted(idx_to_name)                       # ascending column indices
    df = pd.read_csv(path, header=None, usecols=use, quotechar='"',
                     dtype=str, na_values=[""])
    df.columns = [idx_to_name[i] for i in use]      # matches ascending order
    df["season"] = season
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    for c in ["vis_score", "home_score", "attendance", "game_num"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# Reshape to one row per team per game
# ---------------------------------------------------------------------------
def to_team_games(df: pd.DataFrame) -> pd.DataFrame:
    base = ["season", "date", "game_num", "day_night", "park", "attendance"]
    home = df[base].assign(
        team=df["home_team"], opp=df["vis_team"], is_home=1,
        runs_for=df["home_score"], runs_against=df["vis_score"])
    away = df[base].assign(
        team=df["vis_team"], opp=df["home_team"], is_home=0,
        runs_for=df["vis_score"], runs_against=df["home_score"])
    tg = pd.concat([home, away], ignore_index=True)
    tg = tg[tg["team"].isin(DIVISIONS)].copy()       # drop non-standard codes
    tg["win_flag"] = (tg["runs_for"] > tg["runs_against"]).astype(int)
    tg["loss_flag"] = (tg["runs_for"] < tg["runs_against"]).astype(int)
    tg["league"] = tg["team"].map(lambda t: DIVISIONS[t][0])
    tg["division"] = tg["team"].map(lambda t: DIVISIONS[t][1])
    return tg.sort_values(["season", "team", "date", "game_num"]).reset_index(drop=True)


def add_running_record(tg: pd.DataFrame) -> pd.DataFrame:
    grp = [tg["season"], tg["team"]]
    tg["w_after"] = tg["win_flag"].groupby(grp).cumsum()
    tg["l_after"] = tg["loss_flag"].groupby(grp).cumsum()
    tg["w_before"] = tg["w_after"] - tg["win_flag"]   # record ENTERING this game
    tg["l_before"] = tg["l_after"] - tg["loss_flag"]
    tg["team_game_no"] = tg.groupby(["season", "team"]).cumcount() + 1
    return tg


# ---------------------------------------------------------------------------
# Daily standings + games-back
# ---------------------------------------------------------------------------
def daily_standings(tg: pd.DataFrame) -> pd.DataFrame:
    """Cumulative W/L per team as of the END of each calendar date, off-days ffilled."""
    last = (tg.groupby(["season", "team", "date"])
              .agg(w=("w_after", "last"), l=("l_after", "last"))
              .reset_index())
    frames = []
    for season, sdf in last.groupby("season"):
        dates = pd.date_range(sdf["date"].min(), sdf["date"].max(), freq="D")
        for team, tdf in sdf.groupby("team"):
            s = (tdf.set_index("date")[["w", "l"]]
                    .reindex(dates).ffill().fillna(0))
            s.index.name = "date"
            s["season"], s["team"] = season, team
            frames.append(s.reset_index())
    daily = pd.concat(frames, ignore_index=True)
    daily["league"] = daily["team"].map(lambda t: DIVISIONS[t][0])
    daily["division"] = daily["team"].map(lambda t: DIVISIONS[t][1])
    daily["pct"] = daily["w"] / (daily["w"] + daily["l"]).replace(0, np.nan)
    return daily


def _gb(w, l, bw, bl):
    """Games behind a benchmark (bw,bl). Positive = behind, negative = ahead."""
    return ((bw - w) + (l - bl)) / 2.0


def compute_gb(daily: pd.DataFrame) -> pd.DataFrame:
    """
    gb_div      : games behind the team's own division leader (0 if leading)
    gb_wc       : games behind the last wild-card spot (negative if holding one)
    gb_playoff  : distance to nearest playoff entry (<=0 means currently in)

    Approximation: 'division leader' on a given day = best current record in the
    division; ignores head-to-head tiebreakers. Fine as a live competitiveness
    proxy, not as official standings.
    """
    out = []
    for (season, _date), snap in daily.groupby(["season", "date"]):
        snap = snap.copy()
        snap["gb_div"] = np.nan
        snap["gb_wc"] = np.nan
        leaders = {}
        # division GB
        for (lg, dv), d in snap.groupby(["league", "division"]):
            top = d.sort_values(["pct", "w"], ascending=False).iloc[0]
            leaders[(lg, dv)] = top["team"]
            snap.loc[d.index, "gb_div"] = _gb(d["w"], d["l"], top["w"], top["l"])
        # wild-card GB (exclude the three division leaders, rank the rest)
        for lg, d in snap.groupby("league"):
            n = wc_spots(season)
            leader_teams = {leaders[(lg, dv)] for dv in ("E", "C", "W")}
            pool = (d[~d["team"].isin(leader_teams)]
                    .sort_values(["pct", "w"], ascending=False))
            if len(pool) >= n:
                cut = pool.iloc[n - 1]            # last team currently IN a WC spot
                non_leaders = d[~d["team"].isin(leader_teams)].index
                snap.loc[non_leaders, "gb_wc"] = _gb(
                    d.loc[non_leaders, "w"], d.loc[non_leaders, "l"],
                    cut["w"], cut["l"])
        snap["gb_playoff"] = snap[["gb_div", "gb_wc"]].min(axis=1)
        out.append(snap)
    return pd.concat(out, ignore_index=True)


def attach_standings(tg: pd.DataFrame, gb: pd.DataFrame) -> pd.DataFrame:
    """Attach the DAY-BEFORE league context (the state fans see entering the day)."""
    ctx = gb[["season", "team", "date", "gb_div", "gb_wc", "gb_playoff"]].copy()
    ctx["date"] = ctx["date"] + pd.Timedelta(days=1)  # shift so it joins as 'entering'
    return tg.merge(ctx, on=["season", "team", "date"], how="left")


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def build_panel(seasons=SEASONS) -> pd.DataFrame:
    tg = []
    for s in seasons:
        print(f"[{s}] download + parse ...")
        tg.append(to_team_games(parse_gamelog(s)))
    tg = pd.concat(tg, ignore_index=True)
    tg = add_running_record(tg)
    gb = compute_gb(daily_standings(tg))
    panel = attach_standings(tg, gb)

    keep = ["season", "date", "team", "league", "division", "opp", "is_home",
            "day_night", "park", "attendance", "runs_for", "runs_against",
            "win_flag", "team_game_no", "w_before", "l_before",
            "gb_div", "gb_wc", "gb_playoff"]
    panel = (panel[keep]
             .sort_values(["season", "date", "team"])
             .reset_index(drop=True))
    panel.to_csv(OUT_CSV, index=False)
    print(f"wrote {len(panel):,} team-game rows -> {OUT_CSV}")
    return panel


def validate(panel: pd.DataFrame):
    """Two quick sanity checks before you trust a full run."""
    # 1) A known attendance: look up one game you can verify on B-R/MLB.com.
    sample = panel[(panel.is_home == 1) & (panel.season == 2024)].head(1)
    print("\nspot-check one home game's attendance against a known source:")
    print(sample[["date", "team", "opp", "attendance"]].to_string(index=False))

    # 2) Final-day division leaders should sit at gb_div == 0.
    last = panel[panel.season == 2024]
    last = last[last.date == last.date.max()]
    print("\nfinal-day gb_div per team (division winners should be 0):")
    print(last[["team", "division", "gb_div", "gb_playoff"]]
          .sort_values(["division", "gb_div"]).to_string(index=False))


if __name__ == "__main__":
    p = build_panel()
    validate(p)

# ---------------------------------------------------------------------------
# PHASE 2 -- playoff odds (not in this file):
#   Build a daily Monte Carlo: take each day's standings + remaining schedule
#   (also derivable from the game logs), assign each team a strength (prior-year
#   + current-season blend, or a simple Pythagorean-from-run-differential), sim
#   the rest of the season N times, and record each team's playoff frequency.
#   That gives a reproducible daily odds series you fully control -- better for a
#   paper than a black-box tracker, and it dovetails with your point that odds
#   should take over from games-back once they stabilize.
# ---------------------------------------------------------------------------