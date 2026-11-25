"""Scrape r/DeadlockTheGame via the free Arctic Shift API (zero-auth).

Arctic Shift (``https://arctic-shift.photon-reddit.com``) is a public,
no-key archive of Reddit data, so this pipeline runs anywhere with no
credentials or environment variables.

Flow:

1. ``POSTS /api/posts/search`` for ``subreddit=DeadlockTheGame``
   (``limit=100``, ``sort=desc``, ``after=<now - days*86400>``).
2. Rank posts by engagement (``score + num_comments``) and keep the top 25.
3. For each, ``COMMENTS /api/comments/search?link_id=t3_{post_id}&limit=25``
   with a 0.5s delay between calls, and store the comment bodies as a JSON
   array in ``top_comments_json``.
4. Upsert into ``reddit_posts`` (``INSERT OR REPLACE`` — idempotent).

Fallback: if Arctic Shift returns a non-200 status, a clear note is logged
pointing at the Steam discussions hub. Passing ``--steam-fallback`` makes the
scraper also probe ``steamcommunity.com/app/1422450/discussions/0/``
(Deadlock has no Steam user reviews yet; discussions are the closest
community signal). That probe is best-effort and logged only — Steam's
thread shape does not map cleanly onto the Reddit-centric ``reddit_posts``
schema, so nothing is fabricated.
"""

from __future__ import annotations

import json
import logging
import random
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from db import schema
from . import common

ARCTIC_BASE = "https://arctic-shift.photon-reddit.com/api"
POSTS_URL = f"{ARCTIC_BASE}/posts/search"
COMMENTS_URL = f"{ARCTIC_BASE}/comments/search"
STEAM_DISCUSSIONS_URL = "https://steamcommunity.com/app/1422450/discussions/0/"

SUBREDDIT = "DeadlockTheGame"
POSTS_PER_DAY = 12  # sample size per day — spreads coverage across the window
TOP_POSTS = 25
COMMENTS_LIMIT = 25
COMMENT_DELAY = 0.5  # seconds between comment fetches (rate-limit courtesy)


# --------------------------------------------------------------------------- #
# HTTP helpers
# --------------------------------------------------------------------------- #

def _get_json(session, url: str, params: dict, logger, max_attempts: int = 5):
    """GET + parse JSON with backoff.

    Arctic Shift signals throttling with HTTP 422 + an ``error`` body like
    ``"Timeout. Maybe slow down a bit"``, so 422/429/5xx are all treated as
    transient and retried with exponential backoff + jitter. Returns ``None``
    on a permanent 4xx or when retries are exhausted.
    """
    for attempt in range(max_attempts):
        try:
            resp = session.get(url, params=params, timeout=30)
        except requests.RequestException as exc:
            wait = 2.0 * (2 ** attempt) + random.uniform(0, 0.5)
            logger.warning(
                "arctic shift request failed: %s (attempt %d/%d); sleeping %.1fs",
                exc, attempt + 1, max_attempts, wait,
            )
            time.sleep(min(wait, 60.0))
            continue

        if resp.status_code == 200:
            try:
                return resp.json()
            except ValueError as exc:
                logger.warning("arctic shift returned invalid JSON: %s", exc)
                return None

        if resp.status_code in (422, 429) or resp.status_code >= 500:
            msg = ""
            try:
                msg = (resp.json() or {}).get("error") or ""
            except Exception:  # noqa: BLE001
                pass
            wait = 2.0 * (2 ** attempt) + random.uniform(0, 0.5)
            logger.warning(
                "arctic shift %s -> HTTP %s (%s) attempt %d/%d; sleeping %.1fs",
                url, resp.status_code, msg, attempt + 1, max_attempts, wait,
            )
            time.sleep(min(wait, 60.0))
            continue

        # Permanent client error (other 4xx): give up on this call.
        logger.warning("arctic shift %s -> HTTP %s (permanent); skipping", url, resp.status_code)
        return None

    return None


# --------------------------------------------------------------------------- #
# Fetch helpers
# --------------------------------------------------------------------------- #

def fetch_posts(session, days: int, logger) -> list[dict] | None:
    """Return posts sampled across the last ``days`` days (~``POSTS_PER_DAY`` per day).

    A busy subreddit posts 100+ times a day, so a plain ``limit=100`` fetch all
    lands on the same day — which starves the sentiment timeline of history.
    Sampling each day separately spreads coverage across the whole window.
    Returns None if the API is unavailable.
    """
    now = int(time.time())
    posts: list[dict] = []
    seen: set[str] = set()
    for d in range(days):
        day_end = now - d * 86400
        day_start = day_end - 86400
        payload = _get_json(
            session, POSTS_URL,
            {"subreddit": SUBREDDIT, "limit": POSTS_PER_DAY, "sort": "desc",
             "after": day_start, "before": day_end},
            logger,
        )
        if payload is None:
            continue
        for p in payload.get("data") or []:
            pid = p.get("id")
            if pid and pid not in seen:
                seen.add(pid)
                posts.append(p)
        time.sleep(0.4)
    return posts or None


def fetch_comments(session, post_id: str, logger) -> list[dict]:
    """Return up to ``COMMENTS_LIMIT`` comments for a post, sorted by score."""
    payload = _get_json(
        session, COMMENTS_URL,
        {"link_id": f"t3_{post_id}", "limit": COMMENTS_LIMIT},
        logger,
    )
    if payload is None:
        return []
    comments = []
    for c in payload.get("data") or []:
        comments.append(
            {
                "author": c.get("author"),
                "score": int(c.get("score") or 0),
                "body": (c.get("body") or "")[:2000],
                "created_utc": int(c.get("created_utc") or 0),
            }
        )
    comments.sort(key=lambda c: c["score"], reverse=True)
    return comments[:COMMENTS_LIMIT]


# --------------------------------------------------------------------------- #
# Steam fallback (note + optional best-effort probe)
# --------------------------------------------------------------------------- #

def _steam_fallback_note(logger) -> None:
    logger.warning(
        "Arctic Shift unavailable. Manual fallback: %s "
        "(Deadlock has no Steam user reviews; discussion threads are the "
        "closest community signal). Re-run with --steam-fallback to attempt a "
        "best-effort probe of that page.",
        STEAM_DISCUSSIONS_URL,
    )


def _scrape_steam_discussions(session, logger) -> list[dict]:
    """Best-effort probe of the Steam discussions list (fallback only).

    Steam's thread shape (no score / num_comments, different author/date
    fields) does not map cleanly onto ``reddit_posts``, so this logs what it
    finds rather than inserting partial rows.
    """
    resp = common.request_with_retry(session, STEAM_DISCUSSIONS_URL)
    if resp.status_code != 200:
        logger.warning("steam discussions returned HTTP %s; fallback unavailable", resp.status_code)
        return []
    soup = BeautifulSoup(resp.text, "lxml")
    threads = []
    for a in soup.select("a.forum_topic_overlay, a[href*='/discussions/']"):
        title = a.get_text(strip=True)
        href = a.get("href")
        if title and href:
            threads.append({"title": title, "url": href})
    logger.info("steam fallback: found %d discussion threads (not mapped to reddit_posts)", len(threads))
    return threads


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def run(
    db_path: str | Path | None = None,
    logger: logging.Logger | None = None,
    days: int = 7,
    steam_fallback: bool = False,
) -> dict:
    """Scrape posts + comments from Arctic Shift; return row-count summary."""
    logger = logger or logging.getLogger("deadlock_tracker")
    conn = schema.init_db(db_path)
    session = common.get_session()

    posts = fetch_posts(session, days, logger)

    if posts is None:
        _steam_fallback_note(logger)
        if steam_fallback:
            _scrape_steam_discussions(session, logger)
        conn.close()
        return {"reddit_posts_seen": 0, "reddit_posts_inserted": 0}

    if not posts:
        logger.info("arctic shift: no posts in the last %d days", days)
        conn.close()
        return {"reddit_posts_seen": 0, "reddit_posts_inserted": 0}

    ranked = sorted(
        posts,
        key=lambda p: ((p.get("score") or 0) + (p.get("num_comments") or 0), p.get("score") or 0),
        reverse=True,
    )
    top_ids = {p["id"] for p in ranked[:TOP_POSTS] if p.get("id")}

    scraped_at = common.now_iso()
    inserted = 0
    for post in posts:
        post_id = post.get("id")
        if not post_id:
            continue
        # Fetch comments only for the most-engaged posts (keeps the comment
        # crawl bounded); the rest still contribute title/body sentiment.
        if post_id in top_ids:
            comments = fetch_comments(session, post_id, logger)
            time.sleep(COMMENT_DELAY)
        else:
            comments = []

        conn.execute(
            """
            INSERT OR REPLACE INTO reddit_posts
                (post_id, subreddit, title, body, author, score,
                 num_comments, created_utc, top_comments_json, scraped_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                post_id,
                post.get("subreddit") or SUBREDDIT,
                post.get("title"),
                post.get("selftext") or "",
                post.get("author"),
                int(post.get("score") or 0),
                int(post.get("num_comments") or 0),
                int(post.get("created_utc") or 0),
                json.dumps(comments, ensure_ascii=False),
                scraped_at,
            ),
        )
        inserted += 1

    conn.commit()
    conn.close()

    result = {
        "reddit_posts_seen": len(posts),
        "reddit_posts_inserted": inserted,
    }
    logger.info(
        "reddit (arctic shift): %d posts fetched, %d upserted (%d with comments, %d days)",
        len(posts), inserted, len(top_ids), days,
    )
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Scrape r/DeadlockTheGame via Arctic Shift.")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--steam-fallback", action="store_true")
    parser.add_argument("--db", default=None)
    args = parser.parse_args()

    common.setup_logging(verbose=True)
    print(json.dumps(run(db_path=args.db, days=args.days, steam_fallback=args.steam_fallback), indent=2))
