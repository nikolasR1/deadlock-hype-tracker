"""SQLite schema and connection management.

The tracker stores four normalized tables in a single SQLite database with
WAL journal mode and idempotent writes:

  * ``patches``        — one row per patch note.
  * ``patch_changes``  — one row per hero/item/system bullet change inside a patch.
  * ``reddit_posts``   — one row per scraped Reddit post (with top comments).
  * ``hero_stats``     — one row per (date, hero) daily telemetry snapshot.

Every write path uses ``INSERT OR IGNORE`` / ``ON CONFLICT ... DO UPDATE`` so
re-running a scraper is safe and never duplicates rows.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "deadlock.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS patches (
    patch_id     TEXT PRIMARY KEY,
    title        TEXT,
    release_date TEXT,          -- ISO-8601
    url          TEXT,
    raw_text     TEXT,
    scraped_at   TEXT           -- ISO-8601 UTC
);

CREATE TABLE IF NOT EXISTS patch_changes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    patch_id    TEXT NOT NULL REFERENCES patches(patch_id) ON DELETE CASCADE,
    entity_name TEXT,           -- hero or item name (NULL for 'system' bullets)
    entity_type TEXT CHECK (entity_type IN ('hero', 'item', 'system')),
    change_type TEXT CHECK (change_type IN ('buff', 'nerf', 'rework', 'bugfix')),
    summary     TEXT
);
CREATE INDEX IF NOT EXISTS idx_patch_changes_patch ON patch_changes(patch_id);

CREATE TABLE IF NOT EXISTS reddit_posts (
    post_id           TEXT PRIMARY KEY,
    subreddit         TEXT,
    title             TEXT,
    body              TEXT,
    author            TEXT,
    score             INTEGER,
    num_comments      INTEGER,
    created_utc       INTEGER,
    top_comments_json TEXT,     -- JSON array of comment dicts
    scraped_at        TEXT
);

CREATE TABLE IF NOT EXISTS hero_stats (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    date           TEXT,        -- YYYY-MM-DD
    hero_id        INTEGER,
    hero_name      TEXT,
    matches_played INTEGER,
    wins           INTEGER,
    win_rate       REAL,        -- percentage (0-100)
    pick_rate      REAL,        -- percentage of matches featuring the hero
    kda            REAL,
    UNIQUE(date, hero_id)
);

CREATE TABLE IF NOT EXISTS hero_overreaction_metrics (
    computed_at        TEXT,      -- YYYY-MM-DD snapshot date
    hero_name          TEXT,
    mention_count      INTEGER,
    avg_sentiment      REAL,      -- mean VADER compound over hero mentions
    sentiment_z        REAL,      -- z-score of avg_sentiment across heroes
    win_rate_delta     REAL,      -- recent-7d minus prior-7d win rate (pp)
    win_rate_z         REAL,      -- z-score of win_rate_delta across heroes
    overreaction_score REAL,      -- sentiment_z - win_rate_z
    quadrant           TEXT,
    sample_quote       TEXT,      -- most extreme-scoring Reddit mention
    UNIQUE(computed_at, hero_name)
);
"""


def get_connection(db_path: str | Path | None = None) -> sqlite3.Connection:
    """Open a connection, enabling WAL mode and foreign keys.

    Creates the parent directory if it does not exist yet.
    """
    path = Path(db_path) if db_path else DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.execute("PRAGMA busy_timeout=30000;")
    return conn


def init_db(db_path: str | Path | None = None) -> sqlite3.Connection:
    """Create all tables if missing and return a ready connection."""
    conn = get_connection(db_path)
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def table_counts(conn: sqlite3.Connection) -> dict[str, int]:
    """Return a {table: row_count} snapshot for verification/summary."""
    counts = {}
    for table in ("patches", "patch_changes", "reddit_posts", "hero_stats",
                  "hero_overreaction_metrics"):
        row = conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()
        counts[table] = row["n"]
    return counts
