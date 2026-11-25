"""Shared plumbing: logging, HTTP sessions, retry-with-backoff helpers.

All scrapers log to ``logs/pipeline.log`` (structured, timestamped) and to
stdout so both cron output and the local log file stay in sync.
"""

from __future__ import annotations

import logging
import random
import sys
import time
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"
LOG_PATH = LOGS_DIR / "pipeline.log"

DEFAULT_USER_AGENT = (
    "DeadlockHypeTracker/1.0 (personal data-science portfolio; "
    "contact: local only)"
)
DEFAULT_TIMEOUT = 30


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure the package logger once; returns it for convenience."""
    logger = logging.getLogger("deadlock_tracker")
    if logger.handlers:  # already configured
        if verbose:
            logger.setLevel(logging.DEBUG)
        return logger

    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-7s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(LOG_PATH, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    logger.propagate = False
    return logger


def get_session(
    user_agent: str = DEFAULT_USER_AGENT,
    timeout: int = DEFAULT_TIMEOUT,
) -> requests.Session:
    """Build a ``requests.Session`` with a descriptive User-Agent header."""
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": user_agent,
            "Accept": "application/json,text/html,*/*",
            "Accept-Encoding": "gzip, deflate",
        }
    )
    return session


def request_with_retry(
    session: requests.Session,
    url: str,
    *,
    params: dict | None = None,
    headers: dict | None = None,
    max_retries: int = 4,
    base_backoff: float = 2.0,
    logger: logging.Logger | None = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> requests.Response:
    """GET ``url`` with exponential backoff + jitter on 5xx / 429 / 403.

    Returns the final ``Response`` (never raises) so callers can inspect
    ``status_code`` and decide how to degrade. Raises the last transport
    error only if every attempt failed at the network layer.
    """
    log = logger or logging.getLogger("deadlock_tracker")
    last_exc: Exception | None = None
    last_resp: requests.Response | None = None

    for attempt in range(max_retries + 1):
        try:
            resp = session.get(url, params=params, headers=headers, timeout=timeout)

            # Success or a permanent client error (403 block, other 4xx) ->
            # return immediately; retrying a block only wastes cron time.
            if resp.status_code < 400 or (400 <= resp.status_code < 500 and resp.status_code != 429):
                return resp

            # Transient: 429 (rate limit) or 5xx -> honor Retry-After if set,
            # else exponential backoff with jitter.
            retry_after = resp.headers.get("Retry-After")
            if retry_after and retry_after.isdigit() and int(retry_after) > 0:
                wait = float(retry_after)
            else:
                wait = base_backoff * (2 ** attempt) + random.uniform(0, 0.5)
            log.warning(
                "%s returned %s (attempt %d/%d); sleeping %.1fs",
                url, resp.status_code, attempt + 1, max_retries + 1, wait,
            )
            time.sleep(min(wait, 120.0))
            last_resp = resp
            continue

        except requests.RequestException as exc:  # network layer failure
            last_exc = exc
            wait = base_backoff * (2 ** attempt) + random.uniform(0, 0.5)
            log.warning(
                "request to %s failed: %s (attempt %d/%d); sleeping %.1fs",
                url, exc, attempt + 1, max_retries + 1, wait,
            )
            time.sleep(min(wait, 120.0))

    if last_resp is not None:
        return last_resp
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"request_with_retry failed for {url}")


def now_iso() -> str:
    """Current UTC timestamp as ISO-8601 (second precision, 'Z' suffix)."""
    import datetime

    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")
