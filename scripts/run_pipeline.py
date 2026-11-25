"""Unified pipeline runner.

Usage::

    python scripts/run_pipeline.py --source all
    python scripts/run_pipeline.py --source patches --source stats
    python scripts/run_pipeline.py --source reddit

Each source runs in isolation: a failure in one never aborts the others.
At the end a summary table of rows inserted/upserted is printed to stdout
(which doubles as the cron job's delivered output) and to ``logs/pipeline.log``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running as ``python scripts/run_pipeline.py`` from the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from db import schema  # noqa: E402
from scrapers import common  # noqa: E402
from scrapers import fetch_patches, fetch_reddit, fetch_stats  # noqa: E402
from analytics import overreaction  # noqa: E402

SOURCES = ("patches", "reddit", "stats", "analytics")

RUNNERS = {
    "patches": fetch_patches.run,
    "reddit": fetch_reddit.run,
    "stats": fetch_stats.run,
    "analytics": overreaction.run,
}


def _expand(sources: list[str]) -> list[str]:
    if "all" in sources:
        return list(SOURCES)
    ordered = sorted(set(sources))
    # Analytics depends on freshly-scraped reddit posts, so run it last.
    if "analytics" in ordered:
        ordered.remove("analytics")
        ordered.append("analytics")
    return ordered


def _render_table(rows: list[tuple]) -> str:
    """Render a simple aligned table of (source, status, detail)."""
    headers = ("Source", "Status", "Detail")
    rows = [(s, st, d) for (s, st, d) in rows]
    widths = [
        max(len(headers[i]), *(len(r[i]) for r in rows))
        for i in range(len(headers))
    ]
    line = "+".join("-" * (w + 2) for w in widths)
    out = [line]
    out.append(
        "|".join(f" {headers[i]:<{widths[i]}} " for i in range(len(headers)))
    )
    out.append(line)
    for r in rows:
        out.append(
            "|".join(f" {r[i]:<{widths[i]}} " for i in range(len(headers)))
        )
    out.append(line)
    return "\n".join(out)


def _fmt_counts(counts: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in counts.items())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        action="append",
        choices=["all", *SOURCES],
        default=[],
        help="Which scraper(s) to run; repeat for multiple, or 'all'.",
    )
    parser.add_argument(
        "--db",
        default=None,
        help="Override SQLite path (default: data/deadlock.db).",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Reddit scrape window in days (default 7; affects the reddit source only).",
    )
    parser.add_argument(
        "--steam-fallback",
        action="store_true",
        help="If Arctic Shift is down, probe Steam discussions as a fallback.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    if not args.source:
        parser.error("at least one --source is required (all|patches|reddit|stats|analytics)")

    logger = common.setup_logging(verbose=args.verbose)
    targets = _expand(args.source)
    logger.info("pipeline start: sources=%s", targets)

    conn = schema.init_db(args.db)

    table_rows: list[tuple] = []
    for source in targets:
        try:
            if source == "reddit":
                counts = fetch_reddit.run(
                    db_path=args.db,
                    logger=logger,
                    days=args.days,
                    steam_fallback=args.steam_fallback,
                )
            elif source == "analytics":
                counts = overreaction.run(db_path=args.db, logger=logger, days=args.days)
            else:
                counts = RUNNERS[source](db_path=args.db, logger=logger)
            table_rows.append((source, "OK", _fmt_counts(counts)))
        except Exception as exc:  # noqa: BLE001 — isolate per-source failures
            logger.exception("source %r failed", source)
            table_rows.append((source, "FAILED", f"{type(exc).__name__}: {exc}"))

    # Read back final row counts for the summary.
    before = schema.table_counts(conn)
    conn.close()

    summary = _render_table(table_rows)
    print("\n" + summary)
    logger.info("pipeline complete\n%s", summary)

    counts_line = ", ".join(f"{t}={n}" for t, n in before.items())
    print(f"DB row counts -> {counts_line}")
    logger.info("DB row counts -> %s", counts_line)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
