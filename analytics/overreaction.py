"""NLP Overreaction Index engine.

Combines community sentiment (VADER over r/DeadlockTheGame text) with actual
hero win-rate movement (from the Deadlock analytics API) into a per-hero
"overreaction score"::

    overreaction_score = sentiment_z - win_rate_z

Both inputs are standardized (z-scored) across heroes. A large positive score
means the community is far more positive than the hero's actual performance
justifies; a large negative score means the community has soured harder than
the win rate dropped.

Quadrant labels (see ``classify_quadrant``):

    Justified Hype      — positive sentiment (and/or rising win rate)
    Sleeper Buff        — low sentiment but rising win rate (community asleep)
    Placebo Nerf        — low sentiment but flat/positive win rate (nerf was a placebo)
    Legitimately Gutted — low sentiment and a real win-rate drop

Persistence: upserts into ``hero_overreaction_metrics`` (keyed by
``computed_at`` + ``hero_name``) and exports ``data/weekly_digest.json`` with
the top anomalies.
"""

from __future__ import annotations

import datetime
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from db import schema  # noqa: E402
from scrapers import common  # noqa: E402

HERO_STATS_URL = "https://api.deadlock-api.com/v1/analytics/hero-stats"
DIGEST_PATH = schema.DATA_DIR / "weekly_digest.json"

# --------------------------------------------------------------------------- #
# Hero aliases — canonical name -> common short forms for entity matching.
# Only heroes whose name is multi-word / abbreviated / ambiguous need entries;
# single-word names match on the name itself (word-boundary regex).
# --------------------------------------------------------------------------- #
HERO_ALIASES: dict[str, list[str]] = {
    "Mo & Krill": ["mo", "krill", "mnk"],
    "Lady Geist": ["geist"],
    "Grey Talon": ["talon"],
    "Nurse Harrow": ["harrow"],
    "Deadman Danny": ["deadman", "danny"],
    "Rat King": ["ratking"],
    "The Doorman": ["doorman"],
}

# --------------------------------------------------------------------------- #
# Gaming-aware VADER lexicon extension (Deadlock / MOBA slang).
# --------------------------------------------------------------------------- #
LEXICON_EXT: dict[str, float] = {
    "broken": -2.0,
    "gutted": -2.5,
    "unplayable": -2.5,
    "nerfed": -1.5,
    "nerf": -1.5,
    "op": -1.5,
    "buffed": 1.5,
    "buff": 1.0,
    "cracked": 2.0,
    "viable": 1.5,
    "busted": -2.0,
    "garbage": -2.5,
    "trash": -2.0,
    "feeder": -2.0,
    "feed": -1.5,
    "throw": -1.5,
    "throwing": -1.5,
    "useless": -2.0,
    "overpowered": -1.5,
    "clutch": 1.5,
    "pog": 2.0,
    "poggers": 2.0,
    "goat": 2.0,
    "carry": 1.0,
    "melts": 1.0,
    "nuke": 1.0,
    "troll": -1.5,
}

# Quadrant thresholds on the win-rate z-score.
_RISING = 0.25
_DROPPING = -0.25


def get_analyzer() -> SentimentIntensityAnalyzer:
    """Return a VADER analyzer with the gaming lexicon applied."""
    analyzer = SentimentIntensityAnalyzer()
    analyzer.lexicon.update(LEXICON_EXT)
    return analyzer


def classify_quadrant(sentiment_z: float, win_rate_z: float) -> str:
    """Map (sentiment_z, win_rate_z) to a quadrant label.

    Positive sentiment is treated as 'Justified Hype' (the community is not
    overreacting negatively). For low/negative sentiment, the win-rate
    direction decides: clearly dropping -> Legitimately Gutted, clearly rising
    -> Sleeper Buff, otherwise flat -> Placebo Nerf.
    """
    if sentiment_z >= 0:
        return "Justified Hype"
    if win_rate_z <= _DROPPING:
        return "Legitimately Gutted"
    if win_rate_z >= _RISING:
        return "Sleeper Buff"
    return "Placebo Nerf"


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

def load_hero_map(conn) -> dict[int, str]:
    """hero_id -> canonical hero name, from the hero_stats table."""
    rows = conn.execute(
        "SELECT DISTINCT hero_id, hero_name FROM hero_stats WHERE hero_name IS NOT NULL"
    ).fetchall()
    return {r["hero_id"]: r["hero_name"] for r in rows}


def _normalize(text: str) -> str:
    t = text.lower().replace("&", " and ")
    return re.sub(r"\s+", " ", t).strip()


def build_patterns(hero_name: str) -> list[re.Pattern]:
    """Compile word-boundary regexes for a hero's name + aliases (normalized)."""
    base = _normalize(hero_name)
    candidates = [base] + [_normalize(a) for a in HERO_ALIASES.get(hero_name, [])]
    seen = []
    for c in candidates:
        if c and c not in seen:
            seen.append(c)
    return [re.compile(r"\b" + re.escape(c) + r"\b") for c in seen]


def match_heroes(text: str, hero_patterns: dict[str, list[re.Pattern]]) -> set[str]:
    """Return the set of hero names mentioned in ``text``."""
    norm = _normalize(text)
    matched = set()
    for name, pats in hero_patterns.items():
        if any(p.search(norm) for p in pats):
            matched.add(name)
    return matched


def load_reddit_texts(conn, days: int) -> list[dict]:
    """Return [{text, source, post_title}] for posts in the last ``days`` days."""
    cutoff = int(datetime.datetime.now(datetime.timezone.utc).timestamp()) - days * 86400
    rows = conn.execute(
        "SELECT title, body, top_comments_json FROM reddit_posts WHERE created_utc >= ?",
        (cutoff,),
    ).fetchall()

    texts: list[dict] = []
    for r in rows:
        title = (r["title"] or "").strip()
        if title:
            texts.append({"text": title, "source": "title", "post_title": title})
        body = (r["body"] or "").strip()
        if body:
            texts.append({"text": body, "source": "body", "post_title": title})
        try:
            comments = json.loads(r["top_comments_json"] or "[]")
        except (ValueError, TypeError):
            comments = []
        for c in comments:
            cb = (c.get("body") or "").strip()
            if cb:
                texts.append({"text": cb, "source": "comment", "post_title": title})
    return texts


def compute_hero_sentiment(texts, hero_patterns, analyzer) -> dict[str, dict]:
    """Per-hero mention_count / avg_sentiment / sample_quote."""
    hero_scores: dict[str, list[float]] = {n: [] for n in hero_patterns}
    hero_quotes: dict[str, list[tuple[float, str]]] = {n: [] for n in hero_patterns}

    for t in texts:
        matched = match_heroes(t["text"], hero_patterns)
        if not matched:
            continue
        compound = analyzer.polarity_scores(t["text"])["compound"]
        for name in matched:
            hero_scores[name].append(compound)
            hero_quotes[name].append((compound, t["text"]))

    out: dict[str, dict] = {}
    for name, scores in hero_scores.items():
        if not scores:
            out[name] = {"mention_count": 0, "avg_sentiment": 0.0, "sample_quote": ""}
            continue
        extreme = max(hero_quotes[name], key=lambda x: abs(x[0]))
        out[name] = {
            "mention_count": len(scores),
            "avg_sentiment": float(np.mean(scores)),
            "sample_quote": extreme[1][:300],
        }
    return out


def fetch_win_rate_windows(logger, days: int = 7) -> dict[int, dict]:
    """Fetch recent-N-days vs prior-N-days win rates from the Deadlock API.

    Returns ``{hero_id: {win_rate_delta, recent_win_rate, prior_win_rate,
    recent_matches}}``. Raises on failure (caller decides how to degrade).
    """
    now = int(datetime.datetime.now(datetime.timezone.utc).timestamp())
    windows = {
        "recent": {"min": now - days * 86400, "max": now},
        "prior": {"min": now - 2 * days * 86400, "max": now - days * 86400},
    }
    session = common.get_session()

    def rates(win: dict) -> dict[int, dict]:
        resp = common.request_with_retry(
            session, HERO_STATS_URL,
            params={"min_unix_timestamp": win["min"], "max_unix_timestamp": win["max"]},
        )
        resp.raise_for_status()
        out: dict[int, dict] = {}
        for h in resp.json():
            m = int(h.get("matches") or 0)
            w = int(h.get("wins") or 0)
            out[h["hero_id"]] = {
                "matches": m,
                "win_rate": (w / m * 100.0) if m else 0.0,
            }
        return out

    recent = rates(windows["recent"])
    prior = rates(windows["prior"])

    merged: dict[int, dict] = {}
    for hid, r in recent.items():
        p = prior.get(hid, r)  # fall back to recent if a hero is missing in prior
        merged[hid] = {
            "recent_win_rate": r["win_rate"],
            "prior_win_rate": p["win_rate"],
            "win_rate_delta": r["win_rate"] - p["win_rate"],
            "recent_matches": r["matches"],
        }
    return merged


def _zscore(series: pd.Series) -> pd.Series:
    std = series.std()
    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def run(
    db_path: str | Path | None = None,
    logger: logging.Logger | None = None,
    days: int = 7,
) -> dict:
    """Compute overreaction metrics, upsert to DB, export the weekly digest."""
    logger = logger or logging.getLogger("deadlock_tracker")
    conn = schema.init_db(db_path)
    analyzer = get_analyzer()

    hero_map = load_hero_map(conn)
    hero_names = list(dict.fromkeys(hero_map.values()))
    hero_patterns = {n: build_patterns(n) for n in hero_names}

    texts = load_reddit_texts(conn, days)
    sentiment = compute_hero_sentiment(texts, hero_patterns, analyzer)
    logger.info(
        "analytics: %d text units, %d heroes, sentiment over %d-day window",
        len(texts), len(hero_names), days,
    )

    # Win-rate windows (best-effort; degrade to flat deltas if the API is down).
    try:
        win = fetch_win_rate_windows(logger, days=days)
    except Exception as exc:  # noqa: BLE001
        logger.warning("win-rate window fetch failed (%s); using flat deltas", exc)
        win = {}

    rows = []
    for hero_id, name in hero_map.items():
        s = sentiment.get(name, {"mention_count": 0, "avg_sentiment": 0.0, "sample_quote": ""})
        w = win.get(hero_id, {})
        rows.append(
            {
                "hero_name": name,
                "mention_count": int(s["mention_count"]),
                "avg_sentiment": float(s["avg_sentiment"]),
                "sample_quote": s["sample_quote"],
                "win_rate_delta": float(w.get("win_rate_delta", 0.0)),
            }
        )

    df = pd.DataFrame(rows)
    df["sentiment_z"] = _zscore(df["avg_sentiment"])
    df["win_rate_z"] = _zscore(df["win_rate_delta"])
    df["overreaction_score"] = df["sentiment_z"] - df["win_rate_z"]
    df["quadrant"] = df.apply(
        lambda r: classify_quadrant(r["sentiment_z"], r["win_rate_z"]), axis=1
    )

    computed_at = datetime.date.today().isoformat()
    conn.executemany(
        """
        INSERT INTO hero_overreaction_metrics
            (computed_at, hero_name, mention_count, avg_sentiment, sentiment_z,
             win_rate_delta, win_rate_z, overreaction_score, quadrant, sample_quote)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(computed_at, hero_name) DO UPDATE SET
            mention_count      = excluded.mention_count,
            avg_sentiment      = excluded.avg_sentiment,
            sentiment_z        = excluded.sentiment_z,
            win_rate_delta     = excluded.win_rate_delta,
            win_rate_z         = excluded.win_rate_z,
            overreaction_score = excluded.overreaction_score,
            quadrant           = excluded.quadrant,
            sample_quote       = excluded.sample_quote
        """,
        [
            (
                computed_at, r["hero_name"], r["mention_count"], r["avg_sentiment"],
                r["sentiment_z"], r["win_rate_delta"], r["win_rate_z"],
                r["overreaction_score"], r["quadrant"], r["sample_quote"],
            )
            for r in df.to_dict("records")
        ],
    )
    conn.commit()

    # Weekly digest: top 3 anomalies by |overreaction_score| + #1 per quadrant.
    def _record(r):
        return {
            "hero_name": r["hero_name"],
            "quadrant": r["quadrant"],
            "overreaction_score": round(float(r["overreaction_score"]), 4),
            "avg_sentiment": round(float(r["avg_sentiment"]), 4),
            "win_rate_delta": round(float(r["win_rate_delta"]), 4),
            "mention_count": int(r["mention_count"]),
            "sample_quote": r["sample_quote"],
        }

    ranked = df.reindex(df["overreaction_score"].abs().sort_values(ascending=False).index)
    top_anomalies = [_record(r) for r in ranked.head(3).to_dict("records")]

    top_by_quadrant = {}
    for quadrant in ("Placebo Nerf", "Sleeper Buff", "Justified Hype", "Legitimately Gutted"):
        qdf = df[df["quadrant"] == quadrant]
        if not qdf.empty:
            top = qdf.reindex(qdf["overreaction_score"].abs().sort_values(ascending=False).index).iloc[0]
            top_by_quadrant[quadrant] = _record(top)

    digest = {
        "generated_at": common.now_iso(),
        "computed_at": computed_at,
        "top_anomalies": top_anomalies,
        "top_by_quadrant": top_by_quadrant,
    }
    DIGEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    DIGEST_PATH.write_text(json.dumps(digest, indent=2, ensure_ascii=False), encoding="utf-8")

    conn.close()
    result = {
        "hero_overreaction_metrics_upserted": len(df),
        "heroes_with_mentions": int((df["mention_count"] > 0).sum()),
        "computed_at": computed_at,
    }
    logger.info(
        "analytics: upserted %d hero metrics (%d with mentions) for %s",
        len(df), result["heroes_with_mentions"], computed_at,
    )
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute the Deadlock overreaction index.")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--db", default=None)
    args = parser.parse_args()

    common.setup_logging(verbose=True)
    print(json.dumps(run(db_path=args.db, days=args.days), indent=2))
