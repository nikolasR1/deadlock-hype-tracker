"""Scrape Deadlock patch notes and parse structured hero/item changes.

Source: the free community Deadlock API (``https://api.deadlock-api.com``).

* ``GET /v2/patches``          -> patch list with full HTML ``content``.
* ``GET /v1/assets/heroes``    -> hero ``id`` -> canonical ``name``.
* ``GET /v1/assets/items``     -> item names (type ``upgrade`` = buyable items).

The spec's original sources (Liquipedia, deadlock.wiki) are Cloudflare-blocked
or JS-rendered SPAs, so this module uses the API's own patch feed instead,
which carries the full Steam/forum patch text.

Buff/nerf tags are produced by a deterministic keyword heuristic
(``classify_change``); an optional LLM helper (``classify_change_llm``) can
override it when ``LLM_ENDPOINT`` / ``OPENAI_API_KEY`` are configured.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path

from bs4 import BeautifulSoup

from db import schema
from . import common

PATCHES_URL = "https://api.deadlock-api.com/v2/patches"
HEROES_URL = "https://api.deadlock-api.com/v1/assets/heroes"
ITEMS_URL = "https://api.deadlock-api.com/v1/assets/items"

ITEM_CACHE_PATH = schema.DATA_DIR / "item_names.json"
ITEM_CACHE_TTL = 12 * 3600  # seconds; item list changes rarely


# --------------------------------------------------------------------------- #
# HTTP fetch helpers
# --------------------------------------------------------------------------- #

def fetch_patches(session) -> list[dict]:
    """Return the raw patch list from ``/v2/patches``."""
    resp = common.request_with_retry(session, PATCHES_URL)
    resp.raise_for_status()
    return resp.json()


def fetch_hero_names(session) -> dict[int, str]:
    """Map ``hero_id`` -> canonical hero name."""
    resp = common.request_with_retry(session, HEROES_URL)
    resp.raise_for_status()
    return {h["id"]: h["name"] for h in resp.json()}


def fetch_item_names(session, logger) -> set[str]:
    """Return buyable-item names, cached to disk with a TTL.

    The raw items payload is ~6 MB; caching avoids re-downloading it every
    6-hour tick.
    """
    if ITEM_CACHE_PATH.exists():
        age = time.time() - ITEM_CACHE_PATH.stat().st_mtime
        if age < ITEM_CACHE_TTL:
            data = json.loads(ITEM_CACHE_PATH.read_text(encoding="utf-8"))
            logger.debug("loaded %d item names from cache", len(data))
            return set(data)

    resp = common.request_with_retry(session, ITEMS_URL)
    resp.raise_for_status()
    names = sorted(
        {
            it["name"]
            for it in resp.json()
            if it.get("type") == "upgrade" and it.get("name")
        }
    )
    ITEM_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    ITEM_CACHE_PATH.write_text(json.dumps(names), encoding="utf-8")
    logger.info("cached %d item names", len(names))
    return set(names)


# --------------------------------------------------------------------------- #
# Text extraction + change classification
# --------------------------------------------------------------------------- #

def html_to_text(html: str) -> str:
    """Convert patch HTML to readable plain text, one logical line per bullet."""
    if not html:
        return ""
    soup = BeautifulSoup(html, "lxml")
    # Treat <br> and block/list boundaries as line breaks.
    for tag in soup.find_all(["br", "li", "p", "div", "h1", "h2", "h3", "h4"]):
        tag.append("\n")
    text = soup.get_text("\n")
    lines = [ln.strip().lstrip("\u2022-* ").strip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln)


#: Strong signals, checked before generic buff/nerf scoring.
_BUGFIX_TERMS = (
    "fixed", "fix ", "bug", "crash", "issue", "resolved", "corrected",
    "no longer crashes", "hotfix",
)
_REWORK_TERMS = (
    "rework", "reworked", "overhaul", "overhauled", "redesign", "redesigned",
    "replaced", "rebalanced", "rebalance", "revamped", "remade",
)

_BUFF_TERMS = (
    "increased", "increases", "increase", "now also", "now applies", "added",
    "adds", "faster", "improved", "buffed", "stronger", "bonus", "additional",
    "up from", "now grants", "no longer requires", "gains", "gain ", "buffs",
)
_NERF_TERMS = (
    "decreased", "decreases", "decrease", "reduced", "reduces", "reduce",
    "nerfed", "nerfs", "removed", "removes", "slower", "less ", "weaker",
    "down from", "diminish", "cut ", "lowered", "lowers", "more expensive",
)

#: Contextual inversions: "reduced X" is a buff when X is a cost/cooldown.
_BUFF_INVERSIONS = ("reduced cooldown", "lower cooldown", "reduced cost",
                    "reduced price", "cheaper", "reduced damage taken")
_NERF_INVERSIONS = ("increased cooldown", "higher cooldown", "increased cost",
                    "more expensive", "reduced damage", "reduced range",
                    "reduced duration", "reduced radius")


def classify_buff_nerf(text: str) -> str | None:
    """Return 'buff' | 'nerf' | 'rework' | 'bugfix' or None if no clear signal."""
    t = text.lower()

    if any(term in t for term in _BUGFIX_TERMS):
        return "bugfix"
    if any(term in t for term in _REWORK_TERMS):
        return "rework"

    # Unambiguous timing/cost semantics override generic word matching:
    # longer cooldown / cast / reload / cycle time (or higher cost) is a nerf,
    # shorter is a buff — regardless of "increased"/"reduced" word order.
    timing = ("cooldown", "cycle time", "charge time", "reload time", "cast time",
              "channel time")
    if any(w in t for w in timing):
        if any(w in t for w in ("increase", "longer", "higher", "slower")):
            return "nerf"
        if any(w in t for w in ("reduc", "decreas", "lower", "shorten", "faster")):
            return "buff"
    if "cost" in t or "price" in t:
        if any(w in t for w in ("increase", "higher", "more expensive")):
            return "nerf"
        if any(w in t for w in ("reduc", "decreas", "lower", "cheaper")):
            return "buff"

    buff = 0
    nerf = 0
    for term in _BUFF_INVERSIONS:
        if term in t:
            buff += 1
    for term in _NERF_INVERSIONS:
        if term in t:
            nerf += 1
    for term in _BUFF_TERMS:
        if term in t:
            buff += 1
    for term in _NERF_TERMS:
        if term in t:
            nerf += 1

    if buff > nerf:
        return "buff"
    if nerf > buff:
        return "nerf"
    return None  # ambiguous or no signal -> skip (raw text still preserved)


def classify_change(line: str, hero_names: set[str], item_names: set[str]):
    """Parse one bullet line into (entity_name, entity_type, change_type, summary).

    Returns ``None`` when the line carries no confident signal, so
    ``patch_changes`` stays free of noise.
    """
    text = line.strip()
    if len(text) < 4:
        return None

    entity_name: str | None = None
    entity_type: str | None = None
    rest = text

    if ":" in text:
        prefix, rest = text.split(":", 1)
        candidate = prefix.strip()
        # Hero ability bullets are "Hero: Ability - detail" -> keep just the hero.
        candidate = re.split(r"\s*-\s+", candidate, maxsplit=1)[0].strip()
        if candidate in hero_names:
            entity_name, entity_type = candidate, "hero"
        elif candidate in item_names:
            entity_name, entity_type = candidate, "item"
        else:
            entity_name, entity_type = None, "system"
    else:
        entity_name, entity_type = None, "system"

    change_type = classify_buff_nerf(rest)
    if change_type is None:
        return None

    summary = rest.strip() if rest.strip() else text
    return entity_name, entity_type, change_type, summary


def classify_change_llm(line: str) -> str | None:
    """Optional LLM-assisted classifier hook.

    Enabled only when ``LLM_ENDPOINT`` (or ``OPENAI_API_KEY``) is set. Return
    a change_type string, or ``None`` to fall back to the keyword heuristic.
    Kept dependency-free so the pipeline runs with zero external services.
    """
    import os

    if not (os.environ.get("LLM_ENDPOINT") or os.environ.get("OPENAI_API_KEY")):
        return None
    # Intentionally not implemented in the base scaffold; wire your provider
    # here (e.g. POST to an OpenAI-compatible chat endpoint) and return the
    # canonical tag.
    return None


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #

def _insert_patch(conn, patch: dict, raw_text: str, scraped_at: str) -> bool:
    """Insert a patch if unseen; returns True when a NEW row was added."""
    patch_id = (patch.get("guid") or {}).get("text") or patch.get("link") or patch.get("title")
    cur = conn.execute(
        """
        INSERT OR IGNORE INTO patches
            (patch_id, title, release_date, url, raw_text, scraped_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            patch_id,
            patch.get("title"),
            patch.get("pub_date"),
            patch.get("link"),
            raw_text,
            scraped_at,
        ),
    )
    return cur.rowcount == 1


def _insert_changes(conn, patch_id: str, changes: list[tuple], scraped_at: str) -> int:
    """Bulk-insert parsed bullet changes for one patch; returns count."""
    if not changes:
        return 0
    conn.executemany(
        """
        INSERT OR IGNORE INTO patch_changes
            (patch_id, entity_name, entity_type, change_type, summary)
        VALUES (?, ?, ?, ?, ?)
        """,
        [
            (patch_id, name, etype, ctype, summary)
            for (name, etype, ctype, summary) in changes
        ],
    )
    return len(changes)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def run(db_path: str | Path | None = None, logger: logging.Logger | None = None) -> dict:
    """Scrape patches + changes; return row-count summary dict."""
    logger = logger or logging.getLogger("deadlock_tracker")
    conn = schema.init_db(db_path)
    session = common.get_session()
    scraped_at = common.now_iso()

    try:
        patches = fetch_patches(session)
    except Exception as exc:  # noqa: BLE001
        logger.error("patch fetch failed: %s", exc)
        raise

    hero_names = set(fetch_hero_names(session).values())
    item_names = fetch_item_names(session, logger)
    logger.info("loaded %d hero names, %d item names", len(hero_names), len(item_names))

    new_patches = 0
    total_changes = 0
    for patch in patches:
        raw_text = html_to_text(patch.get("content") or "")
        if not raw_text.strip():
            logger.debug("skipping patch with empty content: %r", patch.get("title"))
            continue

        if not _insert_patch(conn, patch, raw_text, scraped_at):
            continue  # already seen; do not re-derive changes

        new_patches += 1
        patch_id = str(
            (patch.get("guid") or {}).get("text")
            or patch.get("link")
            or patch.get("title")
            or "unknown"
        )

        changes: list[tuple] = []
        for line in raw_text.splitlines():
            result = classify_change(line, hero_names, item_names)
            if result is None:
                continue
            name, etype, ctype, summary = result
            # Hero/item bullets keep the whole change; system bullets (e.g.
            # "Fixed a crash...", "Reworked Mid-Boss") are captured too.
            changes.append((name, etype, ctype, summary))
        total_changes += _insert_changes(conn, patch_id, changes, scraped_at)
        logger.debug("patch %r -> %d parsed changes", patch.get("title"), len(changes))

    conn.commit()
    counts = {"patches_inserted": new_patches, "patch_changes_inserted": total_changes}
    logger.info("patches: %d new, %d changes parsed", new_patches, total_changes)
    conn.close()
    return counts


if __name__ == "__main__":
    common.setup_logging(verbose=True)
    print(json.dumps(run(), indent=2))
