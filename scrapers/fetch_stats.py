"""Upsert today's per-hero telemetry snapshot into ``hero_stats``.

Sources (free community Deadlock API):

* ``GET /v1/assets/heroes``          -> ``hero_id`` -> canonical ``name``.
* ``GET /v1/analytics/hero-stats``   -> aggregated per-hero wins / losses /
  matches / kills / deaths / assists.

``win_rate`` and ``pick_rate`` are percentages. ``pick_rate`` is
``hero_matches / total_matches * 100``; because a Deadlock match has 12
players, hero pick rates sum to ~1200% (not 100%) — standard for a MOBA-style
appearance rate. ``kda`` is ``(kills + assists) / deaths``.

Each run overwrites *today's* snapshot via ``ON CONFLICT(date, hero_id) DO
UPDATE``, so the table accumulates a clean daily time series.
"""

from __future__ import annotations

import datetime
import logging
from pathlib import Path

from db import schema
from . import common

HEROES_URL = "https://api.deadlock-api.com/v1/assets/heroes"
HERO_STATS_URL = "https://api.deadlock-api.com/v1/analytics/hero-stats"


def fetch_heroes(session) -> dict[int, str]:
    resp = common.request_with_retry(session, HEROES_URL)
    resp.raise_for_status()
    return {h["id"]: h["name"] for h in resp.json()}


def fetch_hero_stats(session) -> list[dict]:
    resp = common.request_with_retry(session, HERO_STATS_URL)
    resp.raise_for_status()
    return resp.json()


def _safe_rate(numerator: float, denominator: float) -> float:
    return round(numerator / denominator * 100.0, 4) if denominator else 0.0


def run(db_path: str | Path | None = None, logger: logging.Logger | None = None) -> dict:
    """Fetch and upsert today's hero-stats snapshot; return row-count summary."""
    logger = logger or logging.getLogger("deadlock_tracker")
    conn = schema.init_db(db_path)
    session = common.get_session()

    heroes = fetch_heroes(session)
    stats = fetch_hero_stats(session)

    total_matches = sum(int(s.get("matches") or 0) for s in stats)
    today = datetime.date.today().isoformat()

    rows: list[tuple] = []
    for s in stats:
        hero_id = s.get("hero_id")
        if hero_id is None:
            continue
        matches = int(s.get("matches") or 0)
        wins = int(s.get("wins") or 0)
        kills = float(s.get("total_kills") or 0)
        deaths = float(s.get("total_deaths") or 0)
        assists = float(s.get("total_assists") or 0)

        win_rate = _safe_rate(wins, matches)
        pick_rate = _safe_rate(matches, total_matches)
        kda = round((kills + assists) / deaths, 4) if deaths else 0.0

        rows.append(
            (
                today,
                hero_id,
                heroes.get(hero_id, f"hero_{hero_id}"),
                matches,
                wins,
                win_rate,
                pick_rate,
                kda,
            )
        )

    conn.executemany(
        """
        INSERT INTO hero_stats
            (date, hero_id, hero_name, matches_played, wins,
             win_rate, pick_rate, kda)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(date, hero_id) DO UPDATE SET
            hero_name      = excluded.hero_name,
            matches_played = excluded.matches_played,
            wins           = excluded.wins,
            win_rate       = excluded.win_rate,
            pick_rate      = excluded.pick_rate,
            kda            = excluded.kda
        """,
        rows,
    )
    conn.commit()
    conn.close()

    result = {"hero_stats_upserted": len(rows), "date": today}
    logger.info("stats: upserted %d hero rows for %s", len(rows), today)
    return result


if __name__ == "__main__":
    import json

    common.setup_logging(verbose=True)
    print(json.dumps(run(), indent=2))
