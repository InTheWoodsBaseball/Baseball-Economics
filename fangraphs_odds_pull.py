"""
fangraphs_odds_pull.py
------------------------------------------------------------------------------
Pull FanGraphs HISTORICAL daily playoff odds (back to 2016), broken out by round
(Playoffs / Division / LDS / LCS / WS, whatever FG exposes), for every team.

WHY THIS SHAPE (from inspecting your HAR):
  - FanGraphs sits behind Cloudflare -- a plain requests.get() is challenged.
  - There is NO separate odds JSON endpoint and NO _next/data route. The odds are
    server-rendered into the page HTML as a Next.js `__NEXT_DATA__` <script> blob.
  So: fetch the dated page past Cloudflare, pull the blob, parse it. No browser
  needed IF curl_cffi's browser impersonation clears Cloudflare (it usually does).
  If it doesn't, see the note at the bottom -- switch to the Playwright version.

  Your HAR didn't include the main document, so I could NOT see the exact field
  names in the blob. This script AUTO-LOCATES the 30-team odds table and PRINTS
  its columns on the first date. Run it on one season, paste me the printed
  columns, and I'll lock an exact, renamed schema.

URL structure (from your link):
  https://www.fangraphs.com/standings/playoff-odds/{model}/{view}?date=YYYY-MM-DD
  model = 'fg' (the FanGraphs model); view = 'div'. One fetch per date returns all
  teams + all round probabilities in the blob regardless of the view.

Requires:  pip install curl_cffi pandas
Polite + cached: ~200 dates/season, so a full 2016-2025 run is ~2,000 fetches.
Each date's blob is cached to disk, so reruns and re-parses are free. Attribute
FanGraphs on anything published.
------------------------------------------------------------------------------
"""

import gzip
import json
import os
import re
import sys
import time
import datetime as dt

import pandas as pd
from curl_cffi import requests as crequests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
# START WITH ONE SEASON. Confirm the printed columns, then expand to:
#   SEASONS = list(range(2016, 2026))
SEASONS = [2016]

MODEL = "fg"
VIEW = "div"
WINDOW = ("03-20", "10-05")       # month-day span to sweep each season
CACHE_DIR = "fg_odds_cache"
OUT_CSV = "fangraphs_playoff_odds.csv"
SLEEP_SEC = 2.5                   # be polite between live fetches
IMPERSONATE = "chrome"            # curl_cffi browser fingerprint

BASE = "https://www.fangraphs.com/standings/playoff-odds/{model}/{view}?date={date}"


# ---------------------------------------------------------------------------
# Fetch (past Cloudflare) + cache the __NEXT_DATA__ blob per date
# ---------------------------------------------------------------------------
def _cache_path(date_str):
    return os.path.join(CACHE_DIR, f"nextdata_{date_str}.json.gz")


def _looks_like_challenge(html):
    low = html.lower()
    return ("__cf_chl" in low or "just a moment" in low
            or "cf-challenge" in low or "attention required" in low)


def get_next_data(date_str):
    """Return the parsed __NEXT_DATA__ dict for a date, from cache or a live fetch."""
    cp = _cache_path(date_str)
    if os.path.exists(cp):
        with gzip.open(cp, "rt", encoding="utf-8") as f:
            return json.load(f)

    url = BASE.format(model=MODEL, view=VIEW, date=date_str)
    r = crequests.get(url, impersonate=IMPERSONATE, timeout=40)
    html = r.text
    if r.status_code != 200 or _looks_like_challenge(html):
        raise RuntimeError(
            f"Cloudflare blocked {date_str} (status {r.status_code}). "
            "curl_cffi didn't clear the challenge -- switch to the Playwright "
            "variant (ask for it) which renders in a real headless browser.")

    m = re.search(
        r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
        html, re.S)
    if not m:
        raise RuntimeError(f"No __NEXT_DATA__ blob found for {date_str}; page "
                           "layout may have changed.")
    nd = json.loads(m.group(1))

    os.makedirs(CACHE_DIR, exist_ok=True)
    with gzip.open(cp, "wt", encoding="utf-8") as f:
        json.dump(nd, f)
    return nd


# ---------------------------------------------------------------------------
# Auto-locate the team odds table inside the blob
# ---------------------------------------------------------------------------
def _iter_lists(obj):
    if isinstance(obj, list):
        yield obj
        for x in obj:
            yield from _iter_lists(x)
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _iter_lists(v)


def _is_num(v):
    if isinstance(v, (int, float)):
        return True
    if isinstance(v, str):
        s = v.strip().replace(".", "", 1).replace("-", "", 1).replace("%", "", 1)
        return s.isdigit()
    return False


def _score(lst):
    """Heuristic: the odds table is a ~30-row list of dicts with a team-ish key
    and several numeric columns."""
    if not (8 <= len(lst) <= 45) or not all(isinstance(x, dict) for x in lst):
        return -1
    shared = set.intersection(*[set(x.keys()) for x in lst])
    if not shared:
        return -1
    teamish = any("team" in k.lower() or k.lower() in
                  ("abb", "abbname", "shortname", "name") for k in shared)
    numcols = sum(
        1 for k in shared
        if sum(_is_num(x.get(k)) for x in lst) >= max(2, int(0.6 * len(lst))))
    score = len(shared) + 2 * numcols + (10 if teamish else 0)
    if len(lst) == 30:            # exactly 30 teams is a strong signal
        score += 15
    return score


def find_odds_table(next_data):
    best, best_score = None, 0
    for lst in _iter_lists(next_data):
        s = _score(lst)
        if s > best_score:
            best, best_score = lst, s
    return best


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def season_dates(year):
    (m1, d1), (m2, d2) = (WINDOW[0].split("-"), WINDOW[1].split("-"))
    start = dt.date(year, int(m1), int(d1))
    end = dt.date(year, int(m2), int(d2))
    cur = start
    while cur <= end:
        yield cur.isoformat()
        cur += dt.timedelta(days=1)


def build(seasons=SEASONS):
    rows = []
    printed_cols = False
    for year in seasons:
        for date_str in season_dates(year):
            try:
                nd = get_next_data(date_str)
            except RuntimeError as e:
                print(f"  {date_str}: {e}")
                continue
            table = find_odds_table(nd)
            if not table:
                continue          # off-day / preseason / no odds rendered
            if not printed_cols:
                cols = sorted(set.intersection(*[set(r.keys()) for r in table]))
                print("\n>>> ODDS TABLE COLUMNS (paste these back to me):")
                print(cols)
                print(">>> sample row:")
                print(json.dumps(table[0], indent=2)[:1200])
                print()
                printed_cols = True
            for rec in table:
                rec = dict(rec)
                rec["date"] = date_str
                rec["season"] = year
                rows.append(rec)
            # only sleep when we actually hit the network (cache miss already slept)
            print(f"  {date_str}: {len(table)} teams")
            time.sleep(SLEEP_SEC)
    if not rows:
        print("No rows pulled.")
        return None
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nwrote {len(df):,} team-date rows -> {OUT_CSV}")
    return df


if __name__ == "__main__":
    build()

# ---------------------------------------------------------------------------
# If curl_cffi gets Cloudflare-challenged on every date, the fallback is a real
# headless browser. Same parse step -- read __NEXT_DATA__ out of the rendered
# DOM -- just a heavier fetch:
#
#   pip install playwright && playwright install chromium
#   from playwright.sync_api import sync_playwright
#   with sync_playwright() as p:
#       b = p.chromium.launch()
#       pg = b.new_page()
#       pg.goto(url, wait_until="networkidle")
#       html = pg.content()   # then regex __NEXT_DATA__ exactly as above
#
# Ask and I'll wire that version with the same cache + schema.
# ---------------------------------------------------------------------------