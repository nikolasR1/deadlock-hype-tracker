"""Deadlock "Hype vs. Reality" dashboard (Streamlit + Plotly).

Run from the project root:

    streamlit run app.py
"""

from __future__ import annotations

import datetime
import json
import sqlite3
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from analytics.overreaction import build_patterns, get_analyzer, match_heroes

DB_PATH = Path(__file__).resolve().parent / "data" / "deadlock.db"

st.set_page_config(page_title="Deadlock Hype vs. Reality", page_icon="🎮", layout="wide")

QUADRANT_COLORS = {
    "Justified Hype": "#2ecc71",
    "Sleeper Buff": "#3498db",
    "Placebo Nerf": "#f1c40f",
    "Legitimately Gutted": "#e74c3c",
}


# --------------------------------------------------------------------------- #
# Cached DB loaders
# --------------------------------------------------------------------------- #

def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


@st.cache_data(ttl=300, show_spinner=False)
def load_metrics() -> pd.DataFrame:
    conn = _connect()
    df = pd.read_sql(
        """SELECT * FROM hero_overreaction_metrics
           WHERE computed_at = (SELECT MAX(computed_at) FROM hero_overreaction_metrics)""",
        conn,
    )
    conn.close()
    return df


@st.cache_data(ttl=300, show_spinner=False)
def load_reddit() -> pd.DataFrame:
    conn = _connect()
    df = pd.read_sql("SELECT * FROM reddit_posts", conn)
    conn.close()
    return df


@st.cache_data(ttl=300, show_spinner=False)
def load_patches() -> pd.DataFrame:
    conn = _connect()
    df = pd.read_sql("SELECT * FROM patches ORDER BY release_date", conn)
    conn.close()
    return df


@st.cache_data(ttl=300, show_spinner=False)
def load_hero_stats() -> pd.DataFrame:
    conn = _connect()
    df = pd.read_sql("SELECT * FROM hero_stats ORDER BY date", conn)
    conn.close()
    return df


@st.cache_data(ttl=300, show_spinner=False)
def load_patch_changes() -> pd.DataFrame:
    conn = _connect()
    df = pd.read_sql(
        """SELECT pc.entity_name, pc.entity_type, pc.change_type, pc.summary,
                  p.release_date, p.title AS patch_title
           FROM patch_changes pc LEFT JOIN patches p ON pc.patch_id = p.patch_id
           ORDER BY p.release_date DESC""",
        conn,
    )
    conn.close()
    return df


@st.cache_data(ttl=300, show_spinner=False)
def daily_sentiment() -> pd.DataFrame:
    """Average community VADER compound per day (from all reddit text)."""
    reddit = load_reddit()
    analyzer = get_analyzer()
    rows = []
    for _, r in reddit.iterrows():
        texts = [(r["title"] or ""), (r["body"] or "")]
        try:
            comments = json.loads(r["top_comments_json"] or "[]")
        except (ValueError, TypeError):
            comments = []
        texts += [c.get("body") or "" for c in comments]
        scores = [analyzer.polarity_scores(t)["compound"] for t in texts if t and t.strip()]
        if scores:
            day = datetime.datetime.fromtimestamp(
                r["created_utc"], datetime.timezone.utc
            ).strftime("%Y-%m-%d")
            rows.append({"date": day, "sentiment": float(sum(scores) / len(scores))})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.groupby("date", as_index=False)["sentiment"].mean()


def top_in_quadrant(df: pd.DataFrame, quadrant: str):
    q = df[df["quadrant"] == quadrant]
    if q.empty:
        return None
    return q.reindex(q["overreaction_score"].abs().sort_values(ascending=False).index).iloc[0]


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #

def quadrant_chart(metrics: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for quadrant, color in QUADRANT_COLORS.items():
        q = metrics[metrics["quadrant"] == quadrant]
        if q.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=q["sentiment_z"],
                y=q["win_rate_z"],
                mode="markers",
                name=quadrant,
                marker=dict(size=12, color=color, line=dict(width=1, color="#ffffff")),
                text=[
                    f"<b>{n}</b><br>win-rate Δ: {wr:+.2f}pp<br>"
                    f"mentions: {mc}<br><i>{q_[:120]}</i>"
                    for n, wr, mc, q_ in zip(
                        q["hero_name"], q["win_rate_delta"],
                        q["mention_count"], q["sample_quote"],
                    )
                ],
                hovertemplate="%{text}<extra></extra>",
            )
        )
    fig.add_hline(y=0, line=dict(color="#888", dash="dash", width=1))
    fig.add_vline(x=0, line=dict(color="#888", dash="dash", width=1))
    fig.update_layout(
        template="plotly_dark",
        title="Hype vs. Reality — sentiment vs. win-rate movement",
        xaxis_title="sentiment_z (community sentiment →)",
        yaxis_title="win_rate_z (actual performance →)",
        height=560,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def timeline_chart() -> go.Figure:
    sentiment = daily_sentiment()
    hero_stats = load_hero_stats()
    patches = load_patches()

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.update_layout(template="plotly_dark", height=480)

    # Focus on the recent window where data actually lives, instead of the
    # full 2025-2026 patch history (which stretched the axis and hid the dots).
    today = datetime.date.today()
    window_start = today - datetime.timedelta(days=45)
    xmin = pd.Timestamp(window_start)
    xmax = pd.Timestamp(today + datetime.timedelta(days=1))

    if not sentiment.empty:
        sent = sentiment.copy()
        sent["date"] = pd.to_datetime(sent["date"])
        fig.add_trace(
            go.Scatter(
                x=sent["date"], y=sent["sentiment"], mode="lines+markers",
                name="community sentiment",
                line=dict(color="#f1c40f", width=2),
                marker=dict(size=8, color="#f1c40f"),
                hovertemplate="%{x|%b %d}: %{y:+.2f}<extra>community sentiment</extra>",
            ),
            secondary_y=False,
        )

    if not hero_stats.empty:
        daily = hero_stats.groupby("date", as_index=False).agg(win_rate=("win_rate", "mean"))
        daily["date"] = pd.to_datetime(daily["date"])
        fig.add_trace(
            go.Scatter(
                x=daily["date"], y=daily["win_rate"], mode="lines+markers",
                name="avg win rate",
                line=dict(color="#2ecc71", width=2),
                marker=dict(size=8, color="#2ecc71"),
                hovertemplate="%{x|%b %d}: %{y:.2f}%<extra>avg win rate</extra>",
            ),
            secondary_y=True,
        )

    # Patch releases: one short rotated date label per unique release date in
    # the window (instead of every long patch title stacked on top of each
    # other), staggered top/bottom so adjacent dates don't collide.
    patch_dates = []
    if not patches.empty:
        pdates = pd.to_datetime(patches["release_date"], errors="coerce").dt.date
        patch_dates = sorted({d for d in pdates.dropna() if window_start <= d <= today})
    for i, d in enumerate(patch_dates):
        fig.add_vline(
            x=pd.Timestamp(d),
            line=dict(color="#e74c3c", dash="dot", width=1),
            annotation_text=d.strftime("%b %d"),
            annotation_position="top" if i % 2 == 0 else "bottom",
            annotation_textangle=-45,
            annotation_font_size=9,
            annotation_font_color="#e74c3c",
        )

    fig.update_xaxes(range=[xmin, xmax], title_text="date")
    fig.update_yaxes(title_text="sentiment (compound)", secondary_y=False)
    fig.update_yaxes(title_text="win rate (%)", secondary_y=True)
    fig.update_layout(
        title="Patch-to-Sentiment Timeline",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #

st.title("🎮 Deadlock — Hype vs. Reality")
st.caption("Patch & sentiment tracker · SQLite-backed · auto-refreshed by cron")

metrics = load_metrics()
reddit = load_reddit()
patches = load_patches()
hero_stats = load_hero_stats()
patch_changes = load_patch_changes()

# --- KPI row ---------------------------------------------------------------
total_comments = sum(
    len(json.loads(c or "[]")) for c in reddit["top_comments_json"]
) if not reddit.empty else 0

placebo = top_in_quadrant(metrics, "Placebo Nerf") if not metrics.empty else None
sleeper = top_in_quadrant(metrics, "Sleeper Buff") if not metrics.empty else None

c1, c2, c3, c4 = st.columns(4)
c1.metric("Reddit posts analyzed", f"{len(reddit):,}")
c2.metric("Comments analyzed", f"{total_comments:,}")
c3.metric("Patches tracked", f"{len(patches):,}")
c4.metric(
    "#1 Placebo Nerf",
    placebo["hero_name"] if placebo is not None else "—",
    f"{placebo['overreaction_score']:+.2f} score" if placebo is not None else None,
)
c4.metric(
    "#1 Sleeper Buff",
    sleeper["hero_name"] if sleeper is not None else "—",
    f"{sleeper['overreaction_score']:+.2f} score" if sleeper is not None else None,
)

st.divider()

tab_overview, tab_hero = st.tabs(["📈 Overview", "🔍 Hero Deep-Dive"])

with tab_overview:
    if metrics.empty:
        st.info("No overreaction metrics yet — run `python scripts/run_pipeline.py --source analytics`.")
    else:
        st.plotly_chart(quadrant_chart(metrics), use_container_width=True)
    st.plotly_chart(timeline_chart(), use_container_width=True)

    n_days_win = hero_stats["date"].nunique() if not hero_stats.empty else 0
    n_days_reddit = (
        reddit["created_utc"]
        .apply(lambda ts: datetime.datetime.fromtimestamp(ts, datetime.timezone.utc).strftime("%Y-%m-%d"))
        .nunique()
        if not reddit.empty else 0
    )
    if n_days_win <= 2 or n_days_reddit <= 2:
        st.caption(
            "⚠️ Daily history is still accumulating — each scheduled run adds another day, "
            f"so these lines fill in over time (currently {n_days_reddit} day(s) of community data, "
            f"{n_days_win} day(s) of win-rate data)."
        )

with tab_hero:
    hero_names = sorted(metrics["hero_name"].unique()) if not metrics.empty else []
    if not hero_names:
        st.info("No hero data yet.")
    else:
        selected = st.selectbox("Choose a hero", hero_names)

        pats = {n: build_patterns(n) for n in [selected]}
        st.subheader(f"{selected} — recent patch changes")
        changes = patch_changes[
            (patch_changes["entity_name"] == selected)
        ] if not patch_changes.empty else pd.DataFrame()
        if changes.empty:
            st.write("No patch changes on record for this hero.")
        else:
            for _, ch in changes.head(12).iterrows():
                st.markdown(
                    f"`{ch['change_type']}` · {ch['release_date'][:10] if ch['release_date'] else '?'} — {ch['summary']}"
                )

        st.subheader("Win-rate trajectory")
        hs = hero_stats[hero_stats["hero_name"] == selected] if not hero_stats.empty else pd.DataFrame()
        if hs.empty:
            st.write("No win-rate history yet (accumulates with daily runs).")
        else:
            st.dataframe(
                hs[["date", "matches_played", "win_rate", "pick_rate", "kda"]],
                hide_index=True, use_container_width=True,
            )

        st.subheader("Top community mentions")
        mentions = []
        for _, r in reddit.iterrows():
            if match_heroes(r["title"] or "", pats) or match_heroes(r["body"] or "", pats):
                mentions.append((r["score"], f"**post** — {r['title']}"))
            try:
                comments = json.loads(r["top_comments_json"] or "[]")
            except (ValueError, TypeError):
                comments = []
            for cm in comments:
                if match_heroes(cm.get("body") or "", pats):
                    mentions.append((cm.get("score", 0), f"**comment** — {cm.get('body', '')[:200]}"))
        if not mentions:
            st.write("No community mentions found in the current window.")
        else:
            mentions.sort(key=lambda x: x[0], reverse=True)
            for score, txt in mentions[:10]:
                st.markdown(f"[{score} pts] {txt}")
