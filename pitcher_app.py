#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pitcher Dashboard (revised from your v6.0)

Pitcher Dashboard

5) Removed VAA/HAA from pitch metrics + baseline shading + Stuff+ model features
   (features are exactly: velo, dVelo, iVB, dIVB, HB, dHB, spin, arm_angle, extension)

NOTE: If Baseball Savant is down/slow, league pulls can still fail.
This script will catch ReadTimeout and show a Streamlit warning instead of crashing.
"""

import datetime as dt
import time
import unicodedata
from typing import Any, Optional, Tuple, Dict, List

import numpy as np
import pandas as pd

import streamlit as st

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch

import plotly.graph_objects as go

from pybaseball import (
    statcast_pitcher,
    statcast,
    pitching_stats,
    chadwick_register,
)

# Enable pybaseball caching (speeds repeats a lot)
try:
    from pybaseball import cache as pyb_cache
    pyb_cache.enable()
except Exception:
    pass

# Stuff+ modeling
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import HistGradientBoostingRegressor

# For catching timeouts cleanly
try:
    import requests
    _REQ_EXC = (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError)
except Exception:
    _REQ_EXC = (Exception,)

# =========================================================
# IMPORTANT: bump APP_VERSION whenever you change logic
# =========================================================
APP_VERSION = "v6.6"

# =========================================================
# Page config + light CSS
# =========================================================
st.set_page_config(
    page_title="Pitcher Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.0rem; padding-bottom: 1.2rem; }
      div[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
      .stMetric { border-radius: 12px; }
      h1, h2, h3 { margin-bottom: 0.4rem; }
      .tiny { font-size: 0.86rem; color: #6b7280; line-height: 1.25rem; }
      .muted { color: #6b7280; }
      .pill { display:inline-block; padding: 0.18rem 0.55rem; border-radius: 999px; font-size: 0.85rem; }
      .smallnote { font-size: 0.82rem; color: #6b7280; }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# Config
# =========================================================
FASTBALLS = {"FF", "SI", "FT", "FC"}
OFFSPEED = {"CH", "FS", "FO", "SC"}
BREAKING = {"SL", "CU", "KC", "KN", "SV", "CS", "ST"}
PITCH_GROUP_ORDER = ["Fastballs", "Offspeed", "Breaking"]

SWING_DESCRIPTIONS = {
    "swinging_strike", "swinging_strike_blocked", "foul", "foul_tip",
    "hit_into_play", "hit_into_play_no_out", "hit_into_play_score",
    "foul_bunt", "missed_bunt",
}
WHIFF_DESCRIPTIONS = {"swinging_strike", "swinging_strike_blocked"}
STRIKE_DESCRIPTIONS = {
    "called_strike", "swinging_strike", "swinging_strike_blocked",
    "foul", "foul_tip", "foul_bunt", "missed_bunt",
    "hit_into_play", "hit_into_play_no_out", "hit_into_play_score",
}

STRIKEZONE = {"x0": -0.83, "z0": 1.5, "w": 1.66, "h": 2.0}

PITCH_COLORS = {
    "FF": "#d62728",
    "FA": "#d62728",
    "SI": "#ff7f0e",
    "FT": "#ff7f0e",
    "FC": "#bcbd22",
    "CH": "#2ca02c",
    "FS": "#17becf",
    "FO": "#17becf",
    "SC": "#17becf",
    "SL": "#9467bd",
    "ST": "#8c564b",
    "SV": "#8c564b",
    "CU": "#1f77b4",
    "KC": "#1f77b4",
    "KN": "#1f77b4",
    "CS": "#1f77b4",
}

PITCH_NAMES = {
    "FF": "4-Seam Fastball",
    "SI": "Sinker",
    "FT": "2-Seam Fastball",
    "FC": "Cutter",
    "CH": "Changeup",
    "FS": "Splitter",
    "FO": "Forkball",
    "SC": "Screwball",
    "SL": "Slider",
    "ST": "Sweeper",
    "SV": "Slurve",
    "CU": "Curveball",
    "KC": "Knuckle Curve",
    "KN": "Knuckleball",
    "CS": "Slow Curve",
}

INVALID_PITCH_TYPES = {"", "None", "nan", "NaN", "PO"}  # drop pitchouts etc.

MIN_BASELINE_PITCHES = 100

TREND_LABELS = {
    "release_speed": "Velocity",
    "release_spin_rate": "Spin",
    "HB_in": "Horizontal Break (in.)",
    "iVB_in": "Induced Vertical Break (in.)",
    "Ext": "Extension (ft)",
    "arm_angle": "Arm Angle (°)",
    "launch_speed": "Exit Velo",
    "launch_angle": "Launch Angle",
    "estimated_woba_using_speedangle": "xwOBA (contact)",
}

# Stuff+ training defaults
STUFF_TRAIN_YEARS = [2023, 2024, 2025]
STUFF_TRAIN_GAME_TYPES = {"R"}  # train on regular season only
# Keep training fast: stop once we have enough samples per group (across months)
MAX_TRAIN_PITCHES_PER_GROUP = 250_000
TRAIN_SAMPLE_SEED = 42

# =========================================================
# Session-state memo helpers (avoid st.cache_* issues)
# =========================================================
def _ss_get(key: str, default=None):
    return st.session_state.get(key, default)

def _ss_set(key: str, value):
    st.session_state[key] = value

def memo(key: str, builder):
    if key in st.session_state:
        return st.session_state[key]
    v = builder()
    st.session_state[key] = v
    return v

def memo_by_params(prefix: str, params: tuple, builder):
    key = f"{prefix}::{hash(params)}"
    return memo(key, builder)

# =========================================================
# Helpers
# =========================================================
def require_cols(df: pd.DataFrame, cols: list[str]) -> bool:
    return all(c in df.columns for c in cols)

def safe_num(x) -> pd.Series:
    return pd.to_numeric(x, errors="coerce")

def parse_pct(x) -> float | None:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    s = str(x).strip()
    if not s:
        return None
    s = s.replace("%", "")
    v = pd.to_numeric(s, errors="coerce")
    if pd.isna(v):
        return None
    v = float(v)
    if 0 <= v <= 1.0:
        v *= 100.0
    return v

def add_pitch_group(p: str) -> str:
    if p in FASTBALLS:
        return "Fastballs"
    if p in OFFSPEED:
        return "Offspeed"
    if p in BREAKING:
        return "Breaking"
    return "Other"

def adjust_cutter_pitch_group(df: pd.DataFrame, mph_threshold: float = 3.0) -> pd.DataFrame:
    """If a pitcher has both FF and FC, and FC is meaningfully slower (>= threshold mph),
    treat FC as a breaking pitch for grouping purposes (without changing pitch_type).
    This helps cutters that behave like breakers not get lumped with fastballs.
    """
    if df is None or df.empty or not require_cols(df, ["pitcher", "pitch_type", "release_speed", "pitch_group"]):
        return df
    out = df.copy()
    out["pitch_type"] = out["pitch_type"].astype(str)
    # Compute pitcher means
    grp = out.groupby(["pitcher", "pitch_type"], as_index=False)["release_speed"].mean(numeric_only=True)
    ff = grp[grp["pitch_type"] == "FF"].set_index("pitcher")["release_speed"]
    fc = grp[grp["pitch_type"] == "FC"].set_index("pitcher")["release_speed"]
    common = ff.index.intersection(fc.index)
    if len(common) == 0:
        return out
    # FC slower than FF by threshold
    slow = (ff.loc[common] - fc.loc[common]) >= float(mph_threshold)
    pitchers = set(common[slow].tolist())
    if not pitchers:
        return out
    mask = out["pitcher"].isin(list(pitchers)) & out["pitch_type"].eq("FC")
    out.loc[mask, "pitch_group"] = "Breaking"
    return out


def normalize_name(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower().strip()
    s = " ".join(s.split())
    return s

def ip_float_to_outs(ip: float | int | None) -> int | None:
    if ip is None or pd.isna(ip):
        return None
    ipf = float(ip)
    innings = int(np.floor(ipf))
    frac = ipf - innings
    outs_extra = int(round(frac * 10))
    outs_extra = int(np.clip(outs_extra, 0, 2))
    return innings * 3 + outs_extra

def fmt_ip_from_outs(outs: float | int | None) -> str | None:
    if outs is None or pd.isna(outs):
        return None
    outs_i = int(round(float(outs)))
    innings = outs_i // 3
    rem = outs_i % 3
    if rem == 0:
        return f"{innings}"
    if rem == 1:
        return f"{innings} 1/3"
    return f"{innings} 2/3"

def fmt_ip_from_fg(ip_fg: float | int | None) -> str | None:
    outs = ip_float_to_outs(ip_fg)
    return fmt_ip_from_outs(outs) if outs is not None else None

def season_window_statcast(year: int) -> tuple[str, str]:
    if year == 2025:
        return (f"{year}-03-27", f"{year}-09-28")
    return (f"{year}-02-10", f"{year}-12-15")

def allowed_game_types(include_st: bool, include_post: bool) -> set[str]:
    s = {"R"}
    if include_st:
        s.add("S")
    if include_post:
        s.add("P")
    return s

def filter_game_types(df: pd.DataFrame, allowed: set[str]) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if "game_type" not in df.columns:
        return df
    gt = df["game_type"].fillna("").astype(str)
    return df.loc[gt.isin(list(allowed))].copy()

def pitcher_hand(sc: pd.DataFrame) -> str | None:
    if sc is None or sc.empty or "p_throws" not in sc.columns:
        return None
    x = sc["p_throws"].dropna().astype(str)
    if x.empty:
        return None
    return str(x.mode().iloc[0])

def estimated_arm_angle(sc: pd.DataFrame) -> float | None:
    if sc is None or sc.empty or "arm_angle" not in sc.columns:
        return None
    aa = safe_num(sc["arm_angle"]).dropna()
    if aa.empty:
        return None
    return float(aa.median())

def valid_pitch_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "pitch_type" not in df.columns:
        return pd.DataFrame()
    p = df["pitch_type"].astype(str)
    mask = p.notna() & (~p.isin(list(INVALID_PITCH_TYPES)))
    return df.loc[mask].copy()

def month_ranges(start_date: dt.date, end_date: dt.date) -> List[Tuple[dt.date, dt.date]]:
    """Inclusive start, inclusive end (we will pass as strings)."""
    ranges = []
    cur = dt.date(start_date.year, start_date.month, 1)
    while cur <= end_date:
        if cur.month == 12:
            next_month = dt.date(cur.year + 1, 1, 1)
        else:
            next_month = dt.date(cur.year, cur.month + 1, 1)
        month_end = next_month - dt.timedelta(days=1)
        a = max(cur, start_date)
        b = min(month_end, end_date)
        if a <= b:
            ranges.append((a, b))
        cur = next_month
    return ranges

def retry_call(fn, tries: int = 3, base_sleep: float = 1.25):
    for i in range(tries):
        try:
            return fn()
        except _REQ_EXC as e:
            if i == tries - 1:
                raise
            time.sleep(base_sleep * (2 ** i))
        except Exception:
            raise

# =========================================================
# Data loading
# =========================================================
@st.cache_data(ttl=86400, show_spinner=False)
def load_pitcher_dropdown() -> pd.DataFrame:
    """
    Inclusive pitcher list using Chadwick register.
    Keep pitchers + missing/blank position rows (helps fringe/young guys show up).
    """
    def _build():
        reg = chadwick_register().copy()
        reg = reg.dropna(subset=["key_mlbam"]).copy()

        if "position" in reg.columns:
            pos = reg["position"].astype(str)
            reg = reg[(pos.eq("P")) | (pos.eq("nan")) | (pos.eq("None")) | (pos.eq(""))].copy()

        reg["key_mlbam"] = pd.to_numeric(reg["key_mlbam"], errors="coerce").astype("Int64")

        if "key_fangraphs" in reg.columns:
            reg["key_fangraphs"] = pd.to_numeric(reg["key_fangraphs"], errors="coerce").astype("Int64")
        else:
            reg["key_fangraphs"] = pd.Series([pd.NA] * len(reg), dtype="Int64")

        reg = reg.dropna(subset=["key_mlbam"]).copy()

        reg["display"] = (
            reg.get("name_first", "").fillna("").astype(str).str.strip()
            + " "
            + reg.get("name_last", "").fillna("").astype(str).str.strip()
        ).str.strip()
        reg["display_norm"] = reg["display"].map(normalize_name)

        reg = reg[reg["display"].astype(str).str.len() > 0].copy()
        reg = reg.drop_duplicates(subset=["display", "key_mlbam"], keep="first")

        # Filter to only pitchers who appeared in FanGraphs since 2023
        try:
            from pybaseball import pitching_stats
            recent_ids = set()
            recent_names = set()
            for yr in [2023, 2024, 2025]:
                try:
                    df_fg = pitching_stats(yr, qual=0)
                    if df_fg is not None and not df_fg.empty:
                        id_cols = [c for c in ["IDfg", "idfg", "playerid"] if c in df_fg.columns]
                        if id_cols:
                            recent_ids.update(pd.to_numeric(df_fg[id_cols[0]], errors="coerce").dropna().astype(int).tolist())
                        if "Name" in df_fg.columns:
                            recent_names.update(df_fg["Name"].str.lower().str.strip().tolist())
                            # Also store last names for partial matching
                            recent_names.update(df_fg["Name"].str.lower().str.strip().str.split().str[-1].tolist())
                except Exception:
                    pass
            if recent_ids:
                fg_ids = pd.to_numeric(reg["key_fangraphs"], errors="coerce")
                # Keep by FanGraphs ID match OR by name match (catches broken -1 links)
                last_names = reg["display"].str.lower().str.strip().str.split().str[-1]
                name_match = reg["display"].str.lower().str.strip().isin(recent_names) | last_names.isin(recent_names)
                reg = reg[fg_ids.isin(recent_ids) | name_match].copy()
        except Exception:
            pass

        reg = reg.sort_values(["display"]).reset_index(drop=True)
        return reg[["display", "display_norm", "key_mlbam", "key_fangraphs"]]

    return memo("pitcher_dropdown_v66", _build)

def resolve_name_from_mlbam(pitcher_df: pd.DataFrame, mlbam_id: int) -> str:
    hit = pitcher_df.loc[pitcher_df["key_mlbam"].astype("Int64") == int(mlbam_id)]
    if not hit.empty:
        return str(hit.iloc[0]["display"])
    # Fall back to MLB Stats API for players not in Chadwick
    try:
        import requests
        r = requests.get(f"https://statsapi.mlb.com/api/v1/people/{mlbam_id}", timeout=5)
        if r.status_code == 200:
            data = r.json()
            people = data.get("people", [])
            if people:
                return people[0].get("fullName", f"MLBAM {mlbam_id}")
    except Exception:
        pass
    return f"MLBAM {mlbam_id}"

def resolve_fg_from_mlbam(pitcher_df: pd.DataFrame, mlbam_id: int) -> Optional[int]:
    hit = pitcher_df.loc[pitcher_df["key_mlbam"].astype("Int64") == int(mlbam_id)]
    if hit.empty:
        return None
    v = hit.iloc[0].get("key_fangraphs", pd.NA)
    return int(v) if pd.notna(v) else None

def fetch_statcast_pitcher(mlbam_id: int, start_date: str, end_date: str, allowed_gt: set[str]) -> pd.DataFrame:
    def _build():
        df = retry_call(lambda: statcast_pitcher(start_date, end_date, mlbam_id), tries=3)
        df = pd.DataFrame(df) if df is not None else pd.DataFrame()
        df = filter_game_types(df, allowed_gt)
        return df
    return memo_by_params("sc_pitcher_v66", (APP_VERSION, mlbam_id, start_date, end_date, tuple(sorted(list(allowed_gt)))), _build)

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_statcast_league_simple(start_date: str, end_date: str, allowed_gt: frozenset) -> pd.DataFrame:
    df = retry_call(lambda: statcast(start_date, end_date), tries=3)
    df = pd.DataFrame(df) if df is not None else pd.DataFrame()
    df = filter_game_types(df, set(allowed_gt))
    return df

def fetch_statcast_league_chunked(
    start_date: str,
    end_date: str,
    allowed_gt: set[str],
    max_months: int | None = None,
) -> pd.DataFrame:
    """
    Chunk league statcast pull by month to reduce timeouts.
    Uses memoization by (start,end,allowed_gt,max_months).
    """
    def _build():
        sd = dt.datetime.strptime(start_date, "%Y-%m-%d").date()
        ed = dt.datetime.strptime(end_date, "%Y-%m-%d").date()
        ranges = month_ranges(sd, ed)
        if max_months is not None:
            ranges = ranges[:max_months]

        frames = []
        for a, b in ranges:
            a_s = a.strftime("%Y-%m-%d")
            b_s = b.strftime("%Y-%m-%d")
            chunk = retry_call(lambda: statcast(a_s, b_s), tries=3)
            chunk = pd.DataFrame(chunk) if chunk is not None else pd.DataFrame()
            chunk = filter_game_types(chunk, allowed_gt)
            if not chunk.empty:
                frames.append(chunk)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    return memo_by_params(
        "sc_league_chunked_v66",
        (APP_VERSION, start_date, end_date, tuple(sorted(list(allowed_gt))), max_months),
        _build
    )

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fg_pitching_stats_year(year: int) -> pd.DataFrame:
    try:
        df = pitching_stats(year, qual=0)
        return pd.DataFrame(df) if df is not None else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

def fetch_statcast_pitcher_season(mlbam_id: int, year: int, allowed_gt: set[str]) -> pd.DataFrame:
    s, e = season_window_statcast(year)
    return fetch_statcast_pitcher(mlbam_id, s, e, allowed_gt)

def pitcher_first_last_dates(mlbam_id: int, year: int, allowed_gt: set[str]) -> tuple[Optional[dt.date], Optional[dt.date]]:
    sc = fetch_statcast_pitcher_season(mlbam_id, year, allowed_gt)
    if sc is None or sc.empty or "game_date" not in sc.columns:
        return None, None
    g = pd.to_datetime(sc["game_date"], errors="coerce").dropna()
    if g.empty:
        return None, None
    return g.min().date(), g.max().date()

# =========================================================
# Feature engineering
# =========================================================
def add_helpers(sc: pd.DataFrame) -> pd.DataFrame:
    df = sc.copy()

    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")

    if require_cols(df, ["pfx_x", "pfx_z"]):
        df["HB_in"] = -safe_num(df["pfx_x"]) * 12.0
        df["iVB_in"] = safe_num(df["pfx_z"]) * 12.0

    if "release_extension" in df.columns:
        df["Ext"] = safe_num(df["release_extension"])

    if "arm_angle" in df.columns:
        df["arm_angle"] = safe_num(df["arm_angle"])
    else:
        df["arm_angle"] = np.nan

    # VAA, HAA, vRel, hRel
    if all(c in df.columns for c in ["vz0", "vy0", "vx0", "az", "ay", "ax"]):
        vy0 = safe_num(df["vy0"])
        vz0 = safe_num(df["vz0"])
        vx0 = safe_num(df["vx0"])
        ay = safe_num(df["ay"])
        az = safe_num(df["az"])
        ax = safe_num(df["ax"])
        # distance from release to plate ~60.5 - release_pos_y
        t = (-vy0 - np.sqrt(vy0**2 + 2*ay*(17/12))) / ay
        vz_plate = vz0 + az * t
        vy_plate = vy0 + ay * t
        vx_plate = vx0 + ax * t
        df["VAA"] = np.degrees(np.arctan(vz_plate / np.abs(vy_plate)))
        df["HAA"] = np.degrees(np.arctan(vx_plate / np.abs(vy_plate)))
    else:
        df["VAA"] = np.nan
        df["HAA"] = np.nan

    if "release_pos_z" in df.columns:
        df["vRel"] = safe_num(df["release_pos_z"])
    else:
        df["vRel"] = np.nan

    if "release_pos_x" in df.columns:
        df["hRel"] = safe_num(df["release_pos_x"])
    else:
        df["hRel"] = np.nan

    if "description" in df.columns:
        df["is_swing"] = df["description"].isin(SWING_DESCRIPTIONS)
        df["is_whiff"] = df["description"].isin(WHIFF_DESCRIPTIONS)
        df["is_strike_desc"] = df["description"].isin(STRIKE_DESCRIPTIONS)
        df["is_called_strike"] = df["description"].eq("called_strike")
        df["is_swinging_strike"] = df["description"].isin(WHIFF_DESCRIPTIONS)
    else:
        df["is_swing"] = False
        df["is_whiff"] = False
        df["is_strike_desc"] = False
        df["is_called_strike"] = False
        df["is_swinging_strike"] = False

    if "zone" in df.columns:
        df["in_zone"] = safe_num(df["zone"]).between(1, 9)
    else:
        df["in_zone"] = np.nan

    if "stand" not in df.columns:
        df["stand"] = np.nan

    if "pitch_type" in df.columns:
        df["pitch_type"] = df["pitch_type"].astype(str)
        df["pitch_group"] = df["pitch_type"].apply(add_pitch_group)

        df = adjust_cutter_pitch_group(df, mph_threshold=3.0)

    return df

# =========================================================
# xwOBA (Savant-like)
# =========================================================
def xwoba_savant_like(pitches_df: pd.DataFrame) -> float | None:
    if pitches_df is None or pitches_df.empty:
        return None

    # Reset index to avoid pandas 2.x index-alignment errors when called on
    # groupby slices (non-contiguous indices cause TypeError on .loc[mask] assign)
    df = pitches_df.reset_index(drop=True)

    denom = safe_num(df.get("woba_denom", pd.Series(dtype=float)))
    w = safe_num(df.get("woba_value", pd.Series(dtype=float)))
    xw = safe_num(df.get("estimated_woba_using_speedangle", pd.Series(dtype=float)))

    xw_filled = pd.Series(np.where(xw.isna(), w.astype(float), xw.astype(float)), dtype=float)

    ok = (denom > 0) & denom.notna()
    if not ok.any():
        m = xw_filled.mean()
        return None if pd.isna(m) else float(m)

    num = xw_filled[ok].sum()
    d = denom[ok].sum()
    if pd.isna(num) or pd.isna(d) or d == 0:
        return None
    return float(num / d)

# =========================================================
# League pitch-type baselines (for shading) — NO VAA/HAA
# =========================================================
def compute_league_pitchtype_baselines(
    league_sc: pd.DataFrame,
    min_pitches: int = MIN_BASELINE_PITCHES
) -> dict:
    if league_sc is None or league_sc.empty or "pitch_type" not in league_sc.columns:
        return {}

    df = add_helpers(league_sc)
    df = valid_pitch_rows(df)

    metric_cols = [
        ("Velo", "release_speed"),
        ("Spin", "release_spin_rate"),
        ("iVB", "iVB_in"),
        ("HB", "HB_in"),
        ("Ext", "Ext"),
    ]

    for c in ["is_called_strike", "is_swinging_strike", "is_swing", "is_whiff", "in_zone"]:
        if c not in df.columns:
            df[c] = np.nan

    def rate_block(g: pd.DataFrame) -> dict[str, float]:
        pitches = len(g)
        if pitches <= 0:
            return {"CalledStr%": np.nan, "SwStr%": np.nan, "CSW%": np.nan, "Chase%": np.nan, "ZWhiff%": np.nan, "xwOBA": np.nan}

        called = (g["is_called_strike"].sum() / pitches) * 100.0
        csw = ((g["is_called_strike"].sum() + g["is_swinging_strike"].sum()) / pitches) * 100.0

        swings = int(g["is_swing"].sum())
        whiffs = int(g["is_whiff"].sum())
        whiff_pct = (whiffs / swings * 100.0) if swings else np.nan

        in_zone = g["in_zone"].fillna(False).astype(bool)
        z_swings = int((g["is_swing"] & in_zone).sum())
        z_whiffs = int((g["is_whiff"] & in_zone).sum())
        z_miss_pct = (z_whiffs / z_swings * 100.0) if z_swings else np.nan

        out_zone_swings = int((g["is_swing"] & (~in_zone)).sum())
        chase_pct = (out_zone_swings / swings * 100.0) if swings else np.nan

        xwoba = xwoba_savant_like(g)

        return {
            "CalledStr%": float(called) if pd.notna(called) else np.nan,
            "SwStr%": float(whiff_pct) if pd.notna(whiff_pct) else np.nan,
            "CSW%": float(csw) if pd.notna(csw) else np.nan,
            "Chase%": float(chase_pct) if pd.notna(chase_pct) else np.nan,
            "ZWhiff%": float(z_miss_pct) if pd.notna(z_miss_pct) else np.nan,
            "xwOBA": float(xwoba) if xwoba is not None else np.nan,
        }

    baselines: dict[str, dict[str, tuple[float, float]]] = {}

    all_stats: dict[str, tuple[float, float]] = {}
    for out_col, src_col in metric_cols:
        x = safe_num(df.get(src_col, pd.Series(dtype=float)))
        mu = float(x.mean()) if pd.notna(x.mean()) else np.nan
        sd = float(x.std(ddof=0)) if pd.notna(x.std(ddof=0)) else np.nan
        all_stats[out_col] = (mu, sd)

    rates_all = rate_block(df)
    all_stats["CalledStr%"] = (rates_all["CalledStr%"], 6.5)
    all_stats["SwStr%"] = (rates_all["SwStr%"], 10.0)
    all_stats["CSW%"] = (rates_all["CSW%"], 7.5)
    all_stats["Chase%"] = (rates_all["Chase%"], 10.0)
    all_stats["ZWhiff%"] = (rates_all["ZWhiff%"], 10.0)
    all_stats["xwOBA"] = (rates_all["xwOBA"], 0.030)
    all_stats["Stuff+"] = (100.0, 10.0)
    baselines["_ALL_"] = all_stats

    for ptype, g in df.groupby(df["pitch_type"].astype(str), dropna=True):
        if len(g) < min_pitches:
            continue

        stats: dict[str, tuple[float, float]] = {}
        for out_col, src_col in metric_cols:
            x = safe_num(g.get(src_col, pd.Series(dtype=float)))
            mu = float(x.mean()) if pd.notna(x.mean()) else np.nan
            sd = float(x.std(ddof=0)) if pd.notna(x.std(ddof=0)) else np.nan
            stats[out_col] = (mu, sd)

        rates = rate_block(g)
        stats["CalledStr%"] = (rates["CalledStr%"], 6.5)
        stats["SwStr%"] = (rates["SwStr%"], 10.0)
        stats["CSW%"] = (rates["CSW%"], 7.5)
        stats["Chase%"] = (rates["Chase%"], 10.0)
        stats["ZWhiff%"] = (rates["ZWhiff%"], 10.0)
        stats["xwOBA"] = (rates["xwOBA"], 0.030)
        stats["Stuff+"] = (100.0, 10.0)

        baselines[str(ptype)] = stats

    return baselines

# =========================================================
# Zone + Contact block (unchanged)
# =========================================================
def _pa_end_rows(sc: pd.DataFrame) -> pd.DataFrame:
    needed = ["game_pk", "at_bat_number", "pitch_number"]
    if sc is None or sc.empty or not require_cols(sc, needed):
        return pd.DataFrame()
    last_idx = sc.groupby(["game_pk", "at_bat_number"])["pitch_number"].idxmax()
    return sc.loc[last_idx].copy()

def compute_zone_contact_block(sc: pd.DataFrame) -> dict[str, float | None]:
    out: dict[str, float | None] = {
        "Zone%": None,
        "First Pitch Strike%": None,
        "1-1 Strike%": None,
        "AB < 3 Pitches%": None,
        "R2K%": None,
        "Swing%": None,

        "Exit Velo": None,
        "Launch Angle": None,
        "HardHit%": None,
        "Barrel%": None,
        "xSLG (contact)": None,
        "BABIP": None,

        "GB%": None,
        "LD%": None,
        "FB%": None,
        "HR/FB%": None,
        "SweetSpot%": None,
    }
    if sc is None or sc.empty:
        return out

    df = sc.copy()

    if "in_zone" in df.columns and df["in_zone"].notna().any():
        out["Zone%"] = float(df["in_zone"].mean() * 100.0)

    if "is_swing" in df.columns:
        out["Swing%"] = float(df["is_swing"].mean() * 100.0)

    needed = ["game_pk", "at_bat_number", "pitch_number"]
    if require_cols(df, needed) and "is_strike_desc" in df.columns:
        first = (
            df.sort_values(["game_pk", "at_bat_number", "pitch_number"])
              .groupby(["game_pk", "at_bat_number"], as_index=False)
              .first()
        )
        if first["is_strike_desc"].notna().any():
            out["First Pitch Strike%"] = float(first["is_strike_desc"].mean() * 100.0)

    if require_cols(df, ["balls", "strikes", "is_strike_desc"]):
        b = safe_num(df["balls"])
        s = safe_num(df["strikes"])
        m11 = (b == 1) & (s == 1)
        g11 = df.loc[m11]
        if not g11.empty and g11["is_strike_desc"].notna().any():
            out["1-1 Strike%"] = float(g11["is_strike_desc"].mean() * 100.0)

    if require_cols(df, ["game_pk", "at_bat_number", "pitch_number"]):
        end_pitch = df.groupby(["game_pk", "at_bat_number"])["pitch_number"].max()
        if len(end_pitch):
            out["AB < 3 Pitches%"] = float((end_pitch <= 2).mean() * 100.0)
    # R2K%: AB where count reaches 0-2
    if require_cols(df, ["balls", "strikes", "game_pk", "at_bat_number"]):
        df_02 = df[(safe_num(df["balls"]) == 0) & (safe_num(df["strikes"]) == 2)]
        total_abs = df.groupby(["game_pk", "at_bat_number"]).ngroups
        reached_02 = df_02.groupby(["game_pk", "at_bat_number"]).ngroups
        if total_abs > 0:
            out["R2K%"] = float(reached_02 / total_abs * 100.0)

    
    # -----------------------------
    # Batted-ball event (BBE) block
    # -----------------------------
    bbe = df.copy()
    # Only use true batted ball events (bb_type not null) for EV/LA/Barrel
    if "bb_type" in bbe.columns:
        bbe = bbe.loc[bbe["bb_type"].notna()].copy()
    if "launch_speed" in bbe.columns:
        bbe = bbe.loc[bbe["launch_speed"].notna()].copy()
    if "launch_angle" in bbe.columns:
        bbe = bbe.loc[bbe["launch_angle"].notna()].copy()

    if not bbe.empty:
        ev = safe_num(bbe["launch_speed"]).dropna()
        la = safe_num(bbe["launch_angle"]).dropna()

        if len(ev):
            out["Exit Velo"] = float(ev.mean())
            out["HardHit%"] = float((ev >= 95).mean() * 100.0)

        if len(la):
            out["Launch Angle"] = float(la.mean())
            out["SweetSpot%"] = float(((la >= 8) & (la <= 32)).mean() * 100.0)

        # Barrel% (Statcast definition via EV/LA barrel zone)
        def _is_barrel(ev_mph: float, la_deg: float) -> bool:
            if pd.isna(ev_mph) or pd.isna(la_deg) or ev_mph < 98:
                return False
            if ev_mph < 99:
                lo, hi = 26, 30
            elif ev_mph < 100:
                lo, hi = 25, 31
            elif ev_mph < 101:
                lo, hi = 24, 33
            elif ev_mph < 102:
                lo, hi = 23, 34
            elif ev_mph < 103:
                lo, hi = 22, 35
            elif ev_mph < 104:
                lo, hi = 21, 36
            elif ev_mph < 105:
                lo, hi = 20, 37
            elif ev_mph < 106:
                lo, hi = 19, 38
            elif ev_mph < 107:
                lo, hi = 18, 39
            elif ev_mph < 108:
                lo, hi = 17, 40
            elif ev_mph < 109:
                lo, hi = 16, 41
            elif ev_mph < 110:
                lo, hi = 15, 42
            elif ev_mph < 111:
                lo, hi = 14, 43
            elif ev_mph < 112:
                lo, hi = 13, 44
            elif ev_mph < 113:
                lo, hi = 12, 45
            elif ev_mph < 114:
                lo, hi = 11, 46
            elif ev_mph < 115:
                lo, hi = 10, 47
            elif ev_mph < 116:
                lo, hi = 9, 48
            else:
                lo, hi = 8, 50
            return (la_deg >= lo) and (la_deg <= hi)

        ev_series = safe_num(bbe["launch_speed"])
        la_series = safe_num(bbe["launch_angle"])
        barrels = sum(_is_barrel(float(ev_mph), float(la_deg)) for ev_mph, la_deg in zip(ev_series, la_series))
        out["Barrel%"] = float((barrels / len(bbe)) * 100.0) if len(bbe) else None
        import sys; print(f"DEBUG barrel: {barrels} barrels out of {len(bbe)} BBE = {out[chr(66)+chr(97)+chr(114)+chr(114)+chr(101)+chr(108)+chr(37)]:.1f}%", file=sys.stderr)

    if "estimated_slg_using_speedangle" in df.columns:
        xslg = safe_num(df["estimated_slg_using_speedangle"]).dropna()
        if len(xslg):
            out["xSLG (contact)"] = float(xslg.mean())

    if "bb_type" in df.columns:
        bt = df["bb_type"].fillna("").astype(str)
        bip = bt[bt.isin(["ground_ball", "line_drive", "fly_ball", "popup"])]
        denom = len(bip)
        if denom > 0:
            out["GB%"] = float((bip == "ground_ball").sum() / denom * 100.0)
            out["LD%"] = float((bip == "line_drive").sum() / denom * 100.0)
            out["FB%"] = float((bip == "fly_ball").sum() / denom * 100.0)

            if "events" in df.columns:
                hr = int(((df["events"].fillna("").astype(str) == "home_run") & (df.get("bb_type", "") == "fly_ball")).sum())
                fb = int((bip == "fly_ball").sum())
                out["HR/FB%"] = float((hr / fb) * 100.0) if fb > 0 else None

    pa_end = _pa_end_rows(df)
    if not pa_end.empty and "events" in pa_end.columns:
        evs = pa_end["events"].fillna("").astype(str)

        h = int(evs.isin(["single", "double", "triple", "home_run"]).sum())
        hr = int((evs == "home_run").sum())
        so = int(evs.isin(["strikeout", "strikeout_double_play"]).sum())

        non_ab = {"walk", "intent_walk", "hit_by_pitch", "sac_fly", "sac_bunt", "catcher_interf"}
        is_ab = ~evs.isin(list(non_ab)) & evs.ne("")
        ab = int(is_ab.sum())

        denom = (ab - so - hr)
        if denom > 0:
            out["BABIP"] = float((h - hr) / denom)

    return out

# =========================================================
# FanGraphs matching (unchanged)
# =========================================================
def get_fg_row_for_pitcher_year(
    fg_id: int | None,
    mlbam_id: int,
    display_name: str,
    year: int,
) -> dict[str, Any]:
    df = fetch_fg_pitching_stats_year(year)
    if df.empty:
        return {}

    name_cols = [c for c in ["Name", "name"] if c in df.columns]
    if name_cols:
        df["_name_norm"] = df[name_cols[0]].map(normalize_name)
    else:
        df["_name_norm"] = ""

    idfg_cols = [c for c in ["IDfg", "idfg", "playerid"] if c in df.columns]
    mlbid_cols = [c for c in ["MLBID", "mlbid", "MLBId", "mlb_id", "key_mlbam"] if c in df.columns]

    for c in mlbid_cols:
        x = pd.to_numeric(df[c], errors="coerce")
        hit = df[x == float(mlbam_id)]
        if not hit.empty:
            return hit.iloc[0].to_dict()

    if fg_id is not None and pd.notna(fg_id) and idfg_cols:
        c = idfg_cols[0]
        x = pd.to_numeric(df[c], errors="coerce")
        hit = df[x == float(fg_id)]
        if not hit.empty:
            return hit.iloc[0].to_dict()

    dn = normalize_name(display_name)
    hit = df[df["_name_norm"] == dn]
    if not hit.empty:
        return hit.iloc[0].to_dict()

    return {}

def season_xwoba_from_statcast(mlbam_id: int, year: int, allowed_gt: set[str]) -> float | None:
    sc_y = fetch_statcast_pitcher_season(mlbam_id, year, allowed_gt)
    if sc_y is None or sc_y.empty:
        return None
    sc_y = add_helpers(sc_y)
    sc_y = valid_pitch_rows(sc_y)
    return xwoba_savant_like(sc_y)

def build_last_3_seasons_summary(
    fg_id: int | None,
    mlbam_id: int,
    display_name: str,
    current_year: int,
    allowed_gt: set[str],
    include_statcast_xwoba: bool = True,
) -> pd.DataFrame:
    years = [2026, 2025]
    rows = []

    def _num(v):
        return pd.to_numeric(v, errors="coerce")

    for yr in years:
        fg = get_fg_row_for_pitcher_year(fg_id, mlbam_id, display_name, yr)

        ip_fg = _num(fg.get("IP", np.nan))
        era_fg = _num(fg.get("ERA", np.nan))
        fip_fg = _num(fg.get("FIP", np.nan))
        xfip_fg = _num(fg.get("xFIP", np.nan))

        k_pct = parse_pct(fg.get("K%", None))
        bb_pct = parse_pct(fg.get("BB%", None))
        kbb = parse_pct(fg.get("K-BB%", None))

        xw = None
        if include_statcast_xwoba:
            xw = season_xwoba_from_statcast(mlbam_id, yr, allowed_gt={"R"})

        rows.append({
            "Season": yr,
            "IP": fmt_ip_from_fg(ip_fg) if pd.notna(ip_fg) else "—",
            "ERA": float(era_fg) if pd.notna(era_fg) else np.nan,
            "FIP": float(fip_fg) if pd.notna(fip_fg) else np.nan,
            "xFIP": float(xfip_fg) if pd.notna(xfip_fg) else np.nan,
            "K%": float(k_pct) if k_pct is not None else np.nan,
            "BB%": float(bb_pct) if bb_pct is not None else np.nan,
            "K-BB%": float(kbb) if kbb is not None else np.nan,
            "xwOBA": round(float(xw), 3) if xw is not None else np.nan,
        })

    return pd.DataFrame(rows).sort_values("Season", ascending=False).reset_index(drop=True)

# =========================================================
# Styling (red <-> white <-> green) with baseline support
# =========================================================
def _interp_rgb(c0, c1, t: float):
    t = float(np.clip(t, 0.0, 1.0))
    r = int(round(c0[0] + (c1[0] - c0[0]) * t))
    g = int(round(c0[1] + (c1[1] - c0[1]) * t))
    b = int(round(c0[2] + (c1[2] - c0[2]) * t))
    return r, g, b

def style_red_green(
    df: pd.DataFrame,
    directions: dict[str, str],
    fmt_map: dict[str, str] | None = None,
    pitch_col: str | None = None,
    neutral_sd: float = 0.35,
    clip_sd: float = 2.0,
    baselines: dict | None = None,
    baseline_group_col: str | None = None,
    qualify_col: str | None = None,
    qualify_min: int = MIN_BASELINE_PITCHES,
):
    tmp = df.copy()
    sty = tmp.style

    if fmt_map:
        sty = sty.format(fmt_map, na_rep="—")

    green = (64, 160, 92)
    red = (210, 78, 78)
    white = (255, 255, 255)

    table_stats = {}
    for c in directions.keys():
        if c in tmp.columns:
            x = pd.to_numeric(tmp[c], errors="coerce")
            mu = float(x.mean()) if pd.notna(x.mean()) else np.nan
            sd = float(x.std(ddof=0)) if pd.notna(x.std(ddof=0)) else np.nan
            table_stats[c] = (mu, sd)

    def pick_mu_sd(row, col):
        if baselines and baseline_group_col and baseline_group_col in tmp.columns:
            key = str(row.get(baseline_group_col, ""))
            d = baselines.get(key) or baselines.get("_ALL_") or {}
            mu, sd = d.get(col, (np.nan, np.nan))
            if pd.notna(mu) and pd.notna(sd) and sd not in (0, 0.0):
                return mu, sd
        return table_stats.get(col, (np.nan, np.nan))

    def style_cell(val, mu, sd, direction, col_name: str, row_ctx: pd.Series):
        v = pd.to_numeric(val, errors="coerce")
        if pd.isna(v) or pd.isna(mu) or pd.isna(sd) or sd == 0:
            return ""

        z = (float(v) - float(mu)) / float(sd)
        if direction == "low_good":
            z = -z

        if abs(z) <= neutral_sd:
            return "background-color: rgb(255,255,255); color: black;"

        z = float(np.clip(z, -clip_sd, clip_sd))
        t = (z + clip_sd) / (2 * clip_sd)

        if t < 0.5:
            tt = t / 0.5
            r, g, b = _interp_rgb(red, white, tt)
        else:
            tt = (t - 0.5) / 0.5
            r, g, b = _interp_rgb(white, green, tt)

        alpha = 1.0
        if col_name == "Stuff+" and qualify_col and (qualify_col in row_ctx):
            try:
                if pd.notna(row_ctx[qualify_col]) and float(row_ctx[qualify_col]) < float(qualify_min):
                    alpha = 0.25
            except Exception:
                pass
        if alpha < 1.0:
            return f"background-color: rgba({r},{g},{b},{alpha}); color: black;"
        return f"background-color: rgb({r},{g},{b}); color: black;"

    for c, direction in directions.items():
        if c not in tmp.columns:
            continue

        def apply_col(s, col=c, dirn=direction):
            out = []
            for idx, v in s.items():
                row = tmp.loc[idx]
                mu, sd = pick_mu_sd(row, col)
                out.append(style_cell(v, mu, sd, dirn, col_name=col, row_ctx=row))
            return out

        sty = sty.apply(apply_col, subset=[c])

    if pitch_col and pitch_col in tmp.columns:
        name_to_abbrev = {v: k for k, v in PITCH_NAMES.items()}
        def pitch_chip(s: pd.Series):
            out = []
            for v in s.astype(str).tolist():
                abbrev = name_to_abbrev.get(v, v)
                c = PITCH_COLORS.get(abbrev, PITCH_COLORS.get(v, "#9e9e9e"))
                out.append(f"background-color: {c}; color: white; font-weight: 800;")
            return out
        sty = sty.apply(pitch_chip, subset=[pitch_col])

    return sty

def style_usage_delta_table(df: pd.DataFrame, value_cols: list[str]):
    tmp = df.copy()
    sty = tmp.style.format({c: "{:.0f}%" for c in value_cols}, na_rep="—")

    green = (64, 160, 92)
    red = (210, 78, 78)
    white = (255, 255, 255)

    if "All Counts" not in tmp.columns:
        return sty

    base = pd.to_numeric(tmp["All Counts"], errors="coerce")

    for c in value_cols:
        if c not in tmp.columns or c == "All Counts":
            continue

        delta = pd.to_numeric(tmp[c], errors="coerce") - base
        clip = 10.0

        def cell_style(v):
            if pd.isna(v):
                return ""
            z = float(np.clip(v / clip, -1.0, 1.0))
            if abs(z) < 0.15:
                return "background-color: rgb(255,255,255); color: black;"
            t = (z + 1) / 2
            if t < 0.5:
                tt = t / 0.5
                r, g, b = _interp_rgb(red, white, tt)
            else:
                tt = (t - 0.5) / 0.5
                r, g, b = _interp_rgb(white, green, tt)
            return f"background-color: rgb({r},{g},{b}); color: black;"

        sty = sty.apply(lambda s, d=delta: [cell_style(v) for v in d], subset=[c])

    return sty

# =========================================================
# Stuff+ (per-pitch-type, 3 group models)
# Features: velo, dVelo, iVB, dIVB, HB, dHB, spin, arm_angle, extension
# =========================================================
def add_fastball_reference_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reference = pitcher average of ALL fastballs (FF/SI/FT/FC) in the input df.
    Creates:
      fb_velo_ref, fb_spin_ref, fb_ivb_ref, fb_hb_ref
      dVelo, dIVB, dHB
    """
    out = df.copy()
    if out.empty or not require_cols(out, ["pitcher", "pitch_type"]):
        return out

    out = valid_pitch_rows(out)

    refs = []
    for pid, g in out.groupby("pitcher"):
        gfb = g[g["pitch_type"].isin(list(FASTBALLS))]
        if gfb.empty:
            continue
        refs.append({
            "pitcher": pid,
            "fb_velo_ref": safe_num(gfb.get("release_speed", pd.Series(dtype=float))).mean(),
            "fb_spin_ref": safe_num(gfb.get("release_spin_rate", pd.Series(dtype=float))).mean(),
            "fb_ivb_ref": safe_num(gfb.get("iVB_in", pd.Series(dtype=float))).mean(),
            "fb_hb_ref": safe_num(gfb.get("HB_in", pd.Series(dtype=float))).mean(),
        })
    ref_df = pd.DataFrame(refs)
    if ref_df.empty:
        for c in ["fb_velo_ref", "fb_spin_ref", "fb_ivb_ref", "fb_hb_ref", "dVelo", "dIVB", "dHB"]:
            out[c] = np.nan
        return out

    out = out.merge(ref_df, on="pitcher", how="left")
    out["dVelo"] = safe_num(out.get("release_speed", np.nan)) - safe_num(out.get("fb_velo_ref", np.nan))
    out["dIVB"] = safe_num(out.get("iVB_in", np.nan)) - safe_num(out.get("fb_ivb_ref", np.nan))
    out["dHB"] = safe_num(out.get("HB_in", np.nan)) - safe_num(out.get("fb_hb_ref", np.nan))
    return out

def _build_regressor():
    # Nonlinear + fast. Handles interactions better than linear models.
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.08,
        max_depth=6,
        max_iter=300,
        min_samples_leaf=50,
        l2_regularization=0.2,
        random_state=TRAIN_SAMPLE_SEED,
    )



# Stuff+ modeling removed (per app requirements).

def compute_pitch_metrics(sc: pd.DataFrame) -> pd.DataFrame:
    if sc.empty or "pitch_type" not in sc.columns:
        return pd.DataFrame()

    df = sc.copy()
    df = valid_pitch_rows(df)
    total_pitches = len(df)
    if total_pitches == 0:
        return pd.DataFrame()

    def one_block(g: pd.DataFrame) -> dict:
        pitches = len(g)

        velo = safe_num(g.get("release_speed", pd.Series(dtype=float))).mean()
        spin = safe_num(g.get("release_spin_rate", pd.Series(dtype=float))).mean()
        ivb = safe_num(g.get("iVB_in", pd.Series(dtype=float))).mean()
        hb = safe_num(g.get("HB_in", pd.Series(dtype=float))).mean()
        ext = safe_num(g.get("Ext", pd.Series(dtype=float))).mean()
        vaa = safe_num(g.get("VAA", pd.Series(dtype=float))).mean()
        haa = safe_num(g.get("HAA", pd.Series(dtype=float))).mean()
        vrel = safe_num(g.get("vRel", pd.Series(dtype=float))).mean()
        hrel = safe_num(g.get("hRel", pd.Series(dtype=float))).mean()

        called_str_pct = (g["is_called_strike"].sum() / pitches * 100.0) if pitches else np.nan
        csw = ((g["is_called_strike"].sum() + g["is_swinging_strike"].sum()) / pitches * 100.0) if pitches else np.nan

        swstr_pct = (g["is_swinging_strike"].sum() / pitches * 100.0) if pitches else np.nan

        swings = int(g["is_swing"].sum())

        in_zone = g["in_zone"].fillna(False).astype(bool)
        zone_pct = (in_zone.mean() * 100.0) if pitches else np.nan
        z_swings = int((g["is_swing"] & in_zone).sum())
        z_whiffs = int((g["is_whiff"] & in_zone).sum())
        z_miss_pct = (z_whiffs / z_swings * 100.0) if z_swings else np.nan

        out_zone = (~in_zone)
        out_zone_pitches = int(out_zone.sum())
        out_zone_swings = int((g["is_swing"] & out_zone).sum())
        chase_pct = (out_zone_swings / out_zone_pitches * 100.0) if out_zone_pitches else np.nan

        xwoba = xwoba_savant_like(g)

        return {
            "Pitches": int(pitches),
            "Velo": round(float(velo), 1) if pd.notna(velo) else np.nan,
            "iVB": int(round(float(ivb))) if pd.notna(ivb) else np.nan,
            "HB": int(round(float(hb))) if pd.notna(hb) else np.nan,
            "Spin": int(round(float(spin))) if pd.notna(spin) else np.nan,
            "Ext": round(float(ext), 2) if pd.notna(ext) else np.nan,
            "CalledStr%": round(float(called_str_pct), 1) if pd.notna(called_str_pct) else np.nan,
            "SwStr%": round(float(swstr_pct), 1) if pd.notna(swstr_pct) else np.nan,
            "CSW%": round(float(csw), 1) if pd.notna(csw) else np.nan,
            "Chase%": round(float(chase_pct), 1) if pd.notna(chase_pct) else np.nan,
            "Zone%": round(float(zone_pct), 1) if pd.notna(zone_pct) else np.nan,
            "ZWhiff%": round(float(z_miss_pct), 1) if pd.notna(z_miss_pct) else np.nan,
            "xwOBA": round(float(xwoba), 3) if xwoba is not None else np.nan,
            "vRel": round(float(vrel), 1) if pd.notna(vrel) else np.nan,
            "hRel": round(float(hrel), 1) if pd.notna(hrel) else np.nan,
                    }

    rows = []
    for ptype, g in df.groupby("pitch_type", dropna=True):
        pitches = len(g)
        pitch_pct = (pitches / total_pitches * 100.0) if total_pitches else np.nan
        r = {"Pitch": PITCH_NAMES.get(str(ptype), str(ptype)), "Pitch%": round(pitch_pct, 1) if pd.notna(pitch_pct) else np.nan}
        r.update(one_block(g))
        rows.append(r)

    out = pd.DataFrame(rows).sort_values("Pitches", ascending=False).reset_index(drop=True)

    all_row = {"Pitch": "All", "Pitch%": 100.0}
    all_row.update(one_block(df))
    out = pd.concat([out, pd.DataFrame([all_row])], ignore_index=True)

    # order with Stuff+ after xwOBA
    order = [
        "Pitch", "Pitch%", "Pitches",
        "Velo", "iVB", "HB", "Spin", "vRel", "hRel", "Ext",
        "CalledStr%", "SwStr%", "CSW%", "Chase%", "ZWhiff%",
        "xwOBA", "Stuff+",
    ]
    out = out[[c for c in order if c in out.columns]]
    return out

def apply_all_row_mask(pm: pd.DataFrame) -> pd.DataFrame:
    out = pm.copy()
    if out.empty or "Pitch" not in out.columns:
        return out

    allowed = {
        "Pitch", "Pitches",
        "Ext",
        "CalledStr%", "SwStr%", "CSW%", "Chase%", "ZWhiff%",
        "xwOBA",
        "Stuff+",
    }
    mask_all = out["Pitch"].astype(str).eq("All")
    for c in out.columns:
        if c not in allowed and c != "Pitch%" and c != "Pitch":
            out.loc[mask_all, c] = np.nan
    return out

# =========================================================
# Usage table (unchanged)
# =========================================================
def _usage_pct_by_pitch(sc: pd.DataFrame, mask: pd.Series) -> pd.Series:
    df = sc.loc[mask].copy()
    df = valid_pitch_rows(df)
    if df.empty or "pitch_type" not in df.columns:
        return pd.Series(dtype=float)
    counts = (
        df.dropna(subset=["pitch_type"])
          .groupby(df["pitch_type"].astype(str))
          .size()
          .astype(float)
    )
    if counts.sum() <= 0:
        return pd.Series(dtype=float)
    return (counts / counts.sum() * 100.0).sort_values(ascending=False)

def build_usage_situation_table(sc: pd.DataFrame, hand: str) -> pd.DataFrame:
    if sc is None or sc.empty or "stand" not in sc.columns or "pitch_type" not in sc.columns:
        return pd.DataFrame()

    df = sc.copy()
    df = df[df["stand"] == hand].copy()
    df = valid_pitch_rows(df)
    if df.empty:
        return pd.DataFrame()

    all_counts = pd.Series([True] * len(df), index=df.index)

    first_pitch = safe_num(df.get("pitch_number", pd.Series(np.nan, index=df.index))) == 1

    pitch_number = safe_num(df.get("pitch_number", pd.Series(np.nan, index=df.index)))
    strikes = safe_num(df.get("strikes", pd.Series(np.nan, index=df.index)))
    balls = safe_num(df.get("balls", pd.Series(np.nan, index=df.index)))
    early_count = (pitch_number <= 2) & (strikes < 2)

    pitcher_ahead = (strikes > balls)
    pitcher_behind = (balls > strikes)

    pre_two_strikes = (strikes < 2)
    two_strikes = (strikes == 2)

    situations = {
        "All Counts": all_counts,
        "First Pitch": first_pitch,
        "Early Count": early_count,
        "Pitcher Ahead": pitcher_ahead,
        "Pitcher Behind": pitcher_behind,
        "Pre Two Strikes": pre_two_strikes,
        "Two Strikes": two_strikes,
    }

    series_map = {name: _usage_pct_by_pitch(df, m) for name, m in situations.items()}
    pitch_types = sorted(set().union(*[set(s.index.tolist()) for s in series_map.values() if not s.empty]))
    if not pitch_types:
        return pd.DataFrame()

    out = pd.DataFrame(index=pitch_types)
    for name, s in series_map.items():
        out[name] = s.reindex(pitch_types)

    out.insert(0, "Pitch", [PITCH_NAMES.get(p, p) for p in out.index])
    out = out.reset_index(drop=True)
    out = out.sort_values("All Counts", ascending=False, na_position="last").reset_index(drop=True)

    for c in list(situations.keys()):
        out[c] = pd.to_numeric(out[c], errors="coerce").round(0)

    return out

# =========================================================
# Plotting helpers (unchanged)
# =========================================================
def add_strikezone(ax):
    rect = Rectangle(
        (STRIKEZONE["x0"], STRIKEZONE["z0"]),
        STRIKEZONE["w"],
        STRIKEZONE["h"],
        fill=False,
        linewidth=2,
    )
    ax.add_patch(rect)

def add_batter_illustration(ax, hand: str):
    # For catcher POV: LHH stands on right side, RHH on left side
    side = 1.0 if hand == "L" else -1.0
    x0 = 1.6 * side
    y0 = 1.0
    alpha = 0.6
    lw = 2.0
    color = "#444444"

    # Head
    ax.add_patch(Circle((x0, y0 + 1.72), 0.17, fill=False, linewidth=lw, alpha=alpha, color=color))

    # Neck
    ax.plot([x0, x0], [y0 + 1.55, y0 + 1.45], linewidth=lw, alpha=alpha, color=color)

    # Torso - slightly angled toward plate
    torso_top = (x0, y0 + 1.45)
    torso_bot = (x0 - 0.08 * side, y0 + 0.85)
    ax.plot([torso_top[0], torso_bot[0]], [torso_top[1], torso_bot[1]], linewidth=3.0, alpha=alpha, color=color)

    # Hips
    hip_l = (torso_bot[0] - 0.15, y0 + 0.82)
    hip_r = (torso_bot[0] + 0.15, y0 + 0.82)
    ax.plot([hip_l[0], hip_r[0]], [hip_l[1], hip_r[1]], linewidth=2.5, alpha=alpha, color=color)

    # Front leg (toward plate)
    ax.plot([torso_bot[0] - 0.08*side, torso_bot[0] - 0.12*side], [y0 + 0.82, y0 + 0.42], linewidth=lw, alpha=alpha, color=color)
    ax.plot([torso_bot[0] - 0.12*side, torso_bot[0] - 0.10*side], [y0 + 0.42, y0 + 0.0], linewidth=lw, alpha=alpha, color=color)

    # Back leg
    ax.plot([torso_bot[0] + 0.08*side, torso_bot[0] + 0.15*side], [y0 + 0.82, y0 + 0.42], linewidth=lw, alpha=alpha, color=color)
    ax.plot([torso_bot[0] + 0.15*side, torso_bot[0] + 0.18*side], [y0 + 0.42, y0 + 0.0], linewidth=lw, alpha=alpha, color=color)

    # Back arm (top hand on bat)
    ax.plot([torso_top[0], torso_top[0] - 0.15*side], [y0 + 1.30, y0 + 1.50], linewidth=lw, alpha=alpha, color=color)

    # Front arm (bottom hand on bat)  
    ax.plot([torso_top[0], torso_top[0] - 0.20*side], [y0 + 1.20, y0 + 1.45], linewidth=lw, alpha=alpha, color=color)

    # Bat - held up and back in stance
    bat_x = torso_top[0] - 0.18*side
    bat_y = y0 + 1.48
    ax.plot([bat_x, bat_x - 0.15*side], [bat_y, bat_y + 0.55], linewidth=3.5, alpha=0.7, color=color)

def plot_pitch_break_cloud(sc: pd.DataFrame, compact: bool = True):
    needed = ["HB_in", "iVB_in", "pitch_type"]
    if not require_cols(sc, needed):
        st.info("Pitch movement plot needs movement + pitch_type.")
        return

    df = valid_pitch_rows(sc.dropna(subset=needed).copy())
    if df.empty:
        st.info("No pitch movement rows available.")
        return

    ph = pitcher_hand(sc)
    aa = estimated_arm_angle(sc)
    aa_txt = f"{int(round(aa))}°" if aa is not None else "—"

    fig, ax = plt.subplots(figsize=(5.6, 4.8) if compact else (8.2, 6.6))
    ax.grid(True, alpha=0.22)

    for ptype, g in df.groupby("pitch_type"):
        p = str(ptype)
        color = PITCH_COLORS.get(p, "#9e9e9e")
        ax.scatter(
            g["HB_in"], g["iVB_in"],
            s=18 if compact else 30,
            alpha=0.82,
            label=p,
            c=color,
            edgecolors="white",
            linewidths=0.3,
        )

    ax.axhline(0, linewidth=1.1, linestyle="--", alpha=0.5)
    ax.axvline(0, linewidth=1.1, linestyle="--", alpha=0.5)

    ax.set_xlim(-25, 25)
    ax.set_ylim(-25, 25)
    ax.set_aspect("equal", adjustable="box")

    ax.set_title(f"Pitch Breaks — Estimated Arm Angle: {aa_txt}", fontsize=11 if compact else 13)
    ax.set_xlabel("Horizontal Break (in.)")
    ax.set_ylabel("Induced Vertical Break (in.)")

    if aa is not None and np.isfinite(aa):
        theta = np.deg2rad(float(aa))
        L = 22.0
        x_sign = -1.0 if ph == "L" else 1.0
        x_end = x_sign * L * np.cos(theta)
        y_end = L * np.sin(theta)
        ax.plot([0, x_end], [0, y_end], linestyle="--", linewidth=2.0, alpha=0.55)

    ax.legend(
        title="Pitch",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=False,
        fontsize=8 if compact else 9,
        title_fontsize=8 if compact else 9,
    )
    st.pyplot(fig, clear_figure=True)

def plot_heatmap_contour(sc: pd.DataFrame, hand: str, pitch_group: str, mode: str):
    needed = ["plate_x", "plate_z", "stand", "pitch_group"]
    if not require_cols(sc, needed):
        st.info("Heatmaps need plate_x, plate_z, stand, pitch_group.")
        return

    df = sc.dropna(subset=["plate_x", "plate_z", "stand", "pitch_group"]).copy()
    df = df[(df["stand"] == hand) & (df["pitch_group"] == pitch_group)]
    df = valid_pitch_rows(df)
    if df.empty:
        fig, ax = plt.subplots(figsize=(3.6, 3.6))
        add_strikezone(ax)
        add_batter_illustration(ax, hand)
        ax.set_title(f"{pitch_group} vs {hand}HH — {mode}", fontsize=10)
        ax.set_xlabel("Catcher POV")
        ax.set_ylabel("Plate Height")
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-1.0, 5.0)
        ax.text(0, 2.0, "No data", ha="center", va="center", fontsize=11, color="#aaaaaa")
        st.pyplot(fig, clear_figure=True)
        return

    weights = None
    if mode == "Hard Contact (EV)":
        if "launch_speed" not in df.columns:
            st.info("No launch_speed available for EV mode.")
            return
        df = df.dropna(subset=["launch_speed"]).copy()
        if df.empty:
            st.info("No batted-ball EV rows for this slice.")
            return
        weights = safe_num(df["launch_speed"]).fillna(0.0)

    fig, ax = plt.subplots(figsize=(3.6, 3.6))

    if len(df) < 20 or df["plate_x"].nunique() < 3 or df["plate_z"].nunique() < 3:
        ax.scatter(df["plate_x"], df["plate_z"], s=14, alpha=0.35)
    else:
        try:
            sns.kdeplot(
                data=df,
                x="plate_x",
                y="plate_z",
                fill=True,
                levels=9,
                thresh=0.35,
                weights=weights,
                cmap="RdBu_r",
                ax=ax,
            )
        except Exception:
            ax.scatter(df["plate_x"], df["plate_z"], s=14, alpha=0.35)

    add_strikezone(ax)
    add_batter_illustration(ax, hand)

    ax.set_title(f"{pitch_group} vs {hand}HH — {mode}", fontsize=10)
    ax.set_xlabel("Catcher POV")
    ax.set_ylabel("Plate Height")
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-1.0, 5.0)
    st.pyplot(fig, clear_figure=True)

# =========================================================
# Trends (unchanged)
# =========================================================
def _game_opponent_map(sc: pd.DataFrame) -> pd.DataFrame:
    needed = ["game_pk", "game_date", "home_team", "away_team", "inning_topbot"]
    if sc is None or sc.empty or not require_cols(sc, needed):
        return pd.DataFrame(columns=["game_pk", "game_date", "pitcher_team", "opponent"])

    df = sc.dropna(subset=["game_pk", "game_date", "home_team", "away_team", "inning_topbot"]).copy()
    df["inning_topbot"] = df["inning_topbot"].astype(str)
    df["home_team"] = df["home_team"].astype(str)
    df["away_team"] = df["away_team"].astype(str)

    mode_tb = df.groupby("game_pk")["inning_topbot"].agg(lambda s: s.mode().iloc[0] if len(s.mode()) else s.iloc[0])
    base = df.groupby("game_pk", as_index=False).agg({"game_date": "min", "home_team": "first", "away_team": "first"})
    base = base.merge(mode_tb.rename("tb_mode"), on="game_pk", how="left")

    base["pitcher_team"] = np.where(base["tb_mode"].str.lower().str.startswith("top"), base["home_team"], base["away_team"])
    base["opponent"] = np.where(base["pitcher_team"] == base["home_team"], base["away_team"], base["home_team"])

    return base[["game_pk", "game_date", "pitcher_team", "opponent"]]

def trend_by_game(sc: pd.DataFrame, variables: list[str], pitch_filter: str | None) -> pd.DataFrame:
    if sc is None or sc.empty or "game_pk" not in sc.columns or "game_date" not in sc.columns:
        return pd.DataFrame()

    df = sc.dropna(subset=["game_pk", "game_date"]).copy()
    if pitch_filter and pitch_filter != "(All)" and "pitch_type" in df.columns:
        df = df[df["pitch_type"].astype(str) == str(pitch_filter)]

    if df.empty:
        return pd.DataFrame()

    meta = _game_opponent_map(df)

    cols = ["game_pk", "game_date"]
    out = df[cols].copy()
    for v in variables:
        out[v] = safe_num(df[v]) if v in df.columns else np.nan

    g = out.groupby(["game_pk", "game_date"], as_index=False).mean(numeric_only=True)
    g = g.merge(meta, on=["game_pk", "game_date"], how="left")
    return g.sort_values(["game_date", "game_pk"]).reset_index(drop=True)

def plot_trends_plotly(tr: pd.DataFrame, variables: list[str], normalize: bool):
    if tr is None or tr.empty or "game_date" not in tr.columns:
        st.info("No trend data available.")
        return

    plot_df = tr.copy()
    vars_present = [v for v in variables if v in plot_df.columns]
    if not vars_present:
        st.info("None of the selected trend variables are available.")
        return

    if normalize and len(vars_present) > 1:
        for v in vars_present:
            x = safe_num(plot_df[v])
            mu = x.mean()
            sd = x.std()
            plot_df[v] = (x - mu) / sd if (sd and pd.notna(sd) and sd != 0) else np.nan

    opp = plot_df.get("opponent", pd.Series(["—"] * len(plot_df)))
    team = plot_df.get("pitcher_team", pd.Series(["—"] * len(plot_df)))
    gpk = plot_df.get("game_pk", pd.Series([None] * len(plot_df)))

    hover_base = (
        "Date: " + plot_df["game_date"].dt.strftime("%Y-%m-%d").astype(str)
        + "<br>Matchup: " + team.astype(str) + " vs " + opp.astype(str)
        + "<br>GamePK: " + gpk.astype(str)
    )

    fig = go.Figure()
    y_all = []

    for v in vars_present:
        label = TREND_LABELS.get(v, v)
        y = safe_num(plot_df[v])
        y_all.append(y)
        hover = hover_base + "<br>" + label + ": " + y.round(3).astype(str)
        fig.add_trace(go.Scatter(
            x=plot_df["game_date"],
            y=y,
            mode="lines+markers",
            name=label,
            hovertext=hover,
            hoverinfo="text",
        ))

    y_concat = pd.concat(y_all, axis=0).dropna()
    y_range = None
    if len(y_concat):
        y_min = float(y_concat.min())
        y_max = float(y_concat.max())
        span = max(y_max - y_min, 1e-6)
        pad = 0.05 * span
        span = max(y_max - y_min, 1e-6)
        context_pad = max(span * 2.0, 1.0)
        y_range = [y_min - context_pad, y_max + context_pad]

    fig.update_layout(
        height=420,
        margin=dict(l=40, r=20, t=40, b=40),
        xaxis=dict(title="Date", tickformat="%b %d", showgrid=True, tickangle=-25),
        yaxis=dict(
            title="Value" + (" (normalized)" if normalize and len(vars_present) > 1 else ""),
            showgrid=True,
            range=y_range,
            nticks=6,
        ),
        legend=dict(orientation="v", x=1.02, y=1.0),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# App
# =========================================================
def main():
    st.title("PITCHER DASHBOARD")

    pitcher_df = load_pitcher_dropdown()

    usage_cols = {"pitch_usage": "Usage%"}

    # Handle leaderboard jump
    if _ss_get("lb_jump_mlbam"):
        jumped_id = _ss_get("lb_jump_mlbam")
        jumped_name = _ss_get("lb_jump_name", "")
        _ss_set("lb_jump_mlbam", None)
        _ss_set("lb_jump_name", None)
        st.query_params["mlbam"] = str(jumped_id)
        st.info(f"Loading {jumped_name} — enter MLBAM ID {jumped_id} in the Manual MLBAM ID field and click Run / Refresh Data.")

    with st.sidebar:
        st.header("Controls")

        # Search box for dropdown (faster)
        search = st.text_input("Search pitcher", value="", help="Type last name (or part of name) to filter list.")
        filtered = pitcher_df
        if search.strip():
            s = normalize_name(search)
            filtered = pitcher_df[pitcher_df["display_norm"].str.contains(s, na=False)].copy()
            if filtered.empty:
                filtered = pitcher_df

        
        manual_mlbam = st.text_input("Manual MLBAM ID (optional)", value="", help="If a pitcher won’t show / FG ID missing, paste MLBAM id here.")
        manual_id = pd.to_numeric(manual_mlbam, errors="coerce")
        use_manual = pd.notna(manual_id)

        manual_fg = st.text_input("Manual FanGraphs ID (optional)", value="", help="If FanGraphs season stats won’t populate, paste the ID from a FanGraphs URL (e.g., /players/.../32095/...).")
        manual_fg_id = pd.to_numeric(manual_fg, errors="coerce")
        use_manual_fg = pd.notna(manual_fg_id)

        today = dt.date.today()
        default_year = today.year
        season_year = st.selectbox("Season year", options=[2026], index=0)

        if use_manual:
            mlbam_id = int(manual_id)
            display_name = resolve_name_from_mlbam(pitcher_df, mlbam_id)
            fg_id = resolve_fg_from_mlbam(pitcher_df, mlbam_id)
            if use_manual_fg:
                fg_id = int(manual_fg_id)
        else:
            selected_display = st.selectbox("Pitcher", options=filtered["display"].tolist(), index=0)
            row = pitcher_df.loc[pitcher_df["display"] == selected_display].iloc[0]
            mlbam_id = int(row["key_mlbam"])
            fg_id = int(row["key_fangraphs"]) if pd.notna(row.get("key_fangraphs")) else None
            if use_manual_fg:
                fg_id = int(manual_fg_id)
            display_name = selected_display

        st.divider()
        include_st = st.checkbox("Include Spring Training", value=False)
        include_post = st.checkbox("Include Postseason", value=False)
        allowed_gt = allowed_game_types(include_st=include_st, include_post=include_post)

        # Auto-set dates when pitcher/year/GT changes
        auto_key = f"auto_dates::{APP_VERSION}::{mlbam_id}::{season_year}::{include_st}::{include_post}"
        if _ss_get("auto_key") != auto_key:
            s0, e0 = pitcher_first_last_dates(mlbam_id, season_year, allowed_gt=allowed_gt)
            if s0 and e0:
                _ss_set("start_date", s0)
                _ss_set("end_date", e0)
            else:
                _ss_set("start_date", dt.date(season_year, 3, 1))
                _ss_set("end_date", dt.date(season_year, 10, 1))
            _ss_set("auto_key", auto_key)

        start_date = st.date_input("Start date", key="start_date")
        end_date = st.date_input("End date", key="end_date")

        st.divider()
        heat_mode = st.selectbox("Heatmap mode", ["Frequency", "Hard Contact (EV)"])

        st.divider()
        league_compare = st.checkbox(
            f"Compare pitch metrics to league (min {MIN_BASELINE_PITCHES} pitches / pitch type)",
            value=True,
        )
        baseline_days = 30  # kept for compatibility but not shown

        st.divider()

        st.divider()
        include_statcast_xwoba = st.checkbox("Include xwOBA in Season Summary (Statcast)", value=True)

        st.divider()
        run_btn = st.button("Run / Refresh Data", type="primary")

    if start_date > end_date:
        st.error("Start date must be on or before end date.")
        return


    tab_dashboard, tab_leaderboard = st.tabs(["📊 Dashboard", "🏆 Leaderboard"])

    with tab_leaderboard:
        st.markdown("## Pitcher Leaderboard")
        lb_c1, lb_c2 = st.columns(2)
        with lb_c1:
            lb_start = st.date_input("Start date", value=dt.date(dt.date.today().year, 3, 25), key="lb_start")
        with lb_c2:
            lb_end = st.date_input("End date", value=dt.date.today(), key="lb_end")

        lb_hand = st.radio("Pitcher hand", ["All", "RHP", "LHP"], horizontal=True, key="lb_hand")
        lb_pitch = st.selectbox("Pitch type", ["All", "4-Seam Fastball", "Sinker", "Cutter", "Slider", "Sweeper", "Curveball", "Changeup", "Splitter"], key="lb_pitch")
        lb_min = st.slider("Min pitches", 10, 300, 50, key="lb_min")

        PITCH_NAME_TO_CODE = {v: k for k, v in PITCH_NAMES.items()}

        if st.button("Load Leaderboard", key="lb_btn"):
            with st.spinner("Pulling league data..."):
                try:
                    from pybaseball import statcast
                    lg_raw = statcast(lb_start.strftime("%Y-%m-%d"), lb_end.strftime("%Y-%m-%d"))
                    lg_raw = pd.DataFrame(lg_raw) if lg_raw is not None else pd.DataFrame()
                    if not lg_raw.empty:
                        lg_raw = add_helpers(lg_raw)
                        lg_raw = valid_pitch_rows(lg_raw)
                        _ss_set("lb_data", lg_raw)
                    else:
                        st.warning("No data returned for that date range.")
                except Exception as e:
                    st.error(f"Failed: {e}")

        lb_data = _ss_get("lb_data", pd.DataFrame())

        if lb_data is not None and not lb_data.empty:
            df_lb = lb_data.copy()

            # Hand filter
            if lb_hand != "All" and "p_throws" in df_lb.columns:
                hand_code = "R" if lb_hand == "RHP" else "L"
                df_lb = df_lb[df_lb["p_throws"] == hand_code]

            # Pitch filter
            if lb_pitch != "All":
                pt_code = PITCH_NAME_TO_CODE.get(lb_pitch, lb_pitch)
                df_lb = df_lb[df_lb["pitch_type"] == pt_code]

            pitcher_df_lb = load_pitcher_dropdown()
            rows = []
            for pid, g in df_lb.groupby("pitcher"):
                if len(g) < lb_min:
                    continue
                name_hit = pitcher_df_lb.loc[pitcher_df_lb["key_mlbam"].astype("Int64") == int(pid)]
                name = str(name_hit.iloc[0]["display"]) if not name_hit.empty else f"ID {pid}"

                pitches = len(g)
                velo = float(safe_num(g["release_speed"]).mean()) if "release_speed" in g.columns else None
                ivb = float(safe_num(g["iVB_in"]).mean()) if "iVB_in" in g.columns else None
                hb = float(safe_num(g["HB_in"]).mean()) if "HB_in" in g.columns else None
                spin = float(safe_num(g["release_spin_rate"]).mean()) if "release_spin_rate" in g.columns else None
                ext = float(safe_num(g["Ext"]).mean()) if "Ext" in g.columns else None
                swings = int(g["is_swing"].sum()) if "is_swing" in g.columns else 0
                whiffs = int(g["is_whiff"].sum()) if "is_whiff" in g.columns else 0
                called = int(g["is_called_strike"].sum()) if "is_called_strike" in g.columns else 0
                csw = (called + whiffs) / pitches * 100 if pitches else None
                swstr = whiffs / pitches * 100 if pitches else None
                out_zone = (~g["in_zone"].fillna(False).astype(bool))
                ozs = int((g["is_swing"] & out_zone).sum()) if "is_swing" in g.columns else 0
                ozp = int(out_zone.sum())
                chase = ozs / ozp * 100 if ozp else None
                xwoba = xwoba_savant_like(g)

                rows.append({
                    "Pitcher": name,
                    "MLBAM": int(pid),
                    "Pitches": pitches,
                    "Velo": round(velo, 1) if velo else None,
                    "iVB": round(ivb, 1) if ivb else None,
                    "HB": round(hb, 1) if hb else None,
                    "Spin": round(spin, 0) if spin else None,
                    "Ext": round(ext, 2) if ext else None,
                    "CSW%": round(csw, 1) if csw else None,
                    "SwStr%": round(swstr, 1) if swstr else None,
                    "Chase%": round(chase, 1) if chase else None,
                    "xwOBA": round(xwoba, 3) if xwoba else None,
                })

            if rows:
                lb_df = pd.DataFrame(rows)
                sort_col = st.selectbox("Sort by", [c for c in lb_df.columns if c not in ["Pitcher","MLBAM"]], key="lb_sort")
                asc = sort_col in ["xwOBA", "BB%", "ERA", "FIP"]
                lb_df = lb_df.sort_values(sort_col, ascending=asc, na_position="last").reset_index(drop=True)
                lb_df.insert(0, "Rank", range(1, len(lb_df) + 1))
                show_cols = [c for c in lb_df.columns if c != "MLBAM"]
                st.dataframe(lb_df[show_cols], use_container_width=True, hide_index=True)

                st.markdown("---")
                st.caption("Select a pitcher to load into the Dashboard:")
                sel = st.selectbox("Pitcher", lb_df["Pitcher"].tolist(), key="lb_sel", label_visibility="collapsed")
                if st.button("→ Load into Dashboard", key="lb_view", type="primary"):
                    mlbam_sel = int(lb_df[lb_df["Pitcher"]==sel]["MLBAM"].iloc[0])
                    _ss_set("lb_jump_mlbam", mlbam_sel)
                    _ss_set("lb_jump_name", sel)
                    st.rerun()
            else:
                st.info("No pitchers met the minimum pitch threshold.")
        else:
            st.info("Set a date range and click Load Leaderboard to begin.")

    with tab_dashboard:
        # ── Yesterday's Notables Ticker ──────────────────────────────────
        try:
            import requests as _req
            yesterday = (dt.date.today() - dt.timedelta(days=1)).strftime("%Y-%m-%d")
            sched = _req.get(f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={yesterday}&hydrate=decisions", timeout=8).json()
            yday_dates = sched.get("dates", [])
            notables = []
            if yday_dates:
                for game in yday_dates[0].get("games", []):
                    game_pk = game.get("gamePk")
                    try:
                        box = _req.get(f"https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore", timeout=8).json()
                        for side in ["away", "home"]:
                            team_data = box.get("teams", {}).get(side, {})
                            pitchers = team_data.get("pitchers", [])
                            players = team_data.get("players", {})
                            if not pitchers:
                                continue
                            starter_id = pitchers[0]
                            p = players.get(f"ID{starter_id}", {})
                            stats = p.get("stats", {}).get("pitching", {})
                            if stats.get("gamesStarted", 0) == 0:
                                continue
                            name = p.get("person", {}).get("fullName", "")
                            team_abbr = team_data.get("team", {}).get("abbreviation", "")
                            ip = stats.get("inningsPitched", "0")
                            k = stats.get("strikeOuts", 0)
                            er = stats.get("earnedRuns", 0)
                            pitches = stats.get("pitchesThrown", 0)
                            # Build notable string
                            parts = [f"{float(ip):.1f} IP"]
                            if k >= 8: parts.append(f"{k} K")
                            if er == 0 and float(ip) >= 5: parts.append("0 ER")
                            notables.append(f"**{name}** ({team_abbr}) — {' · '.join(parts)}")
                    except Exception:
                        continue
            if notables:
                ticker = "   |   ".join(notables)
                st.markdown(f"<div style='background:#f0f4ff;padding:6px 12px;border-radius:8px;font-size:0.85rem;margin-bottom:8px'>📅 <b>Yesterday ({yesterday}):</b>   {ticker}</div>", unsafe_allow_html=True)
        except Exception:
            pass

        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")
        current_year = int(end_date.year)

        params = (
            APP_VERSION, mlbam_id, fg_id, display_name, start_str, end_str,
            heat_mode, league_compare, baseline_days, include_statcast_xwoba,
            include_st, include_post
        )

        if run_btn or _ss_get("loaded_params") != params:
            # --- Pitcher pull ---
            with st.spinner("Fetching Statcast pitcher data..."):
                try:
                    sc_raw = fetch_statcast_pitcher(mlbam_id, start_str, end_str, allowed_gt=allowed_gt)
                except _REQ_EXC:
                    st.error("Statcast pitcher pull timed out. Try a smaller date range and rerun.")
                    return
                except Exception as e:
                    st.error(f"Statcast pitcher pull failed: {e}")
                    return

            if sc_raw.empty:
                _ss_set("sc", pd.DataFrame())
                _ss_set("baselines", {})
                _ss_set("league_zone_contact", {})
                _ss_set("loaded_params", params)
                st.warning("No Statcast rows returned. Try a wider date range (or enable ST/Postseason).")
                return

            sc = add_helpers(sc_raw)
            _ss_set("sc", sc)

            baselines = {}
            league_zone_contact = {}

            # --- League pulls (baseline window) ---
            if league_compare:
                # Use full season dates for league baseline
                base_start_str = f"{end_date.year}-03-01"
                base_end_str = end_date.strftime("%Y-%m-%d")

                with st.spinner("Fetching league Statcast (season baseline)..."):
                    try:
                        lg = fetch_statcast_league_simple(base_start_str, base_end_str, allowed_gt=frozenset(allowed_gt))
                    except _REQ_EXC:
                        lg = pd.DataFrame()
                        st.warning("League pull timed out (baseline window). Pitch shading + Stuff+ may be unavailable.")
                    except Exception:
                        lg = pd.DataFrame()
                        st.warning("League pull failed (baseline window). Pitch shading + Stuff+ may be unavailable.")

                if lg is not None and not lg.empty:
                    lg = add_helpers(lg)
                    baselines = compute_league_pitchtype_baselines(lg, min_pitches=MIN_BASELINE_PITCHES)
                    league_zone_contact = compute_zone_contact_block(lg)

            _ss_set("baselines", baselines)
            _ss_set("league_zone_contact", league_zone_contact)
            _ss_set("loaded_params", params)

        sc: pd.DataFrame = _ss_get("sc", pd.DataFrame())
        baselines: dict = _ss_get("baselines", {})
        league_zone_contact: dict = _ss_get("league_zone_contact", {})

        if sc is None or sc.empty:
            st.info("Open the sidebar and click **Run / Refresh Data**.")
            return

        # -----------------------------------------------------
        # Season summary
        # -----------------------------------------------------
        with st.spinner("Building last-3-seasons summary..."):
            season_tbl = build_last_3_seasons_summary(
                fg_id=fg_id,
                mlbam_id=mlbam_id,
                display_name=display_name,
                current_year=current_year,
                allowed_gt={"R"},
                include_statcast_xwoba=include_statcast_xwoba,
            )
        # -----------------------------------------------------
        # Pitch metrics (fixed totals)
        # -----------------------------------------------------
        # Compute per-pitch-type usage% per game and add to sc
        # usage_cols defined at top of main as {"pitch_usage": "Usage%"}

        pitch_metrics = compute_pitch_metrics(sc)
        pitch_metrics_disp = apply_all_row_mask(pitch_metrics)

        # -----------------------------------------------------
        # Header + season summary
        # -----------------------------------------------------
        # Team logo + headshot
        TEAM_IDS = {
            "ARI":109,"ATL":144,"BAL":110,"BOS":111,"CHC":112,"CWS":145,"CIN":113,
            "CLE":114,"COL":115,"DET":116,"HOU":117,"KC":118,"LAA":108,"LAD":119,
            "MIA":146,"MIL":158,"MIN":142,"NYM":121,"NYY":147,"OAK":133,"PHI":143,
            "PIT":134,"SD":135,"SEA":136,"SF":137,"STL":138,"TB":139,"TEX":140,
            "TOR":141,"WSH":120,
        }

        pitcher_team = None
        if not sc.empty and "home_team" in sc.columns and "away_team" in sc.columns and "inning_topbot" in sc.columns:
            tb = sc["inning_topbot"].fillna("").astype(str).mode()
            tb = tb.iloc[0] if not tb.empty else ""
            if tb.lower().startswith("top"):
                pitcher_team = sc["home_team"].dropna().astype(str).mode().iloc[0] if not sc["home_team"].dropna().empty else None
            else:
                pitcher_team = sc["away_team"].dropna().astype(str).mode().iloc[0] if not sc["away_team"].dropna().empty else None

        headshot_url = f"https://midfield.mlbstatic.com/v1/people/{mlbam_id}/spots/spot"
        team_id = TEAM_IDS.get(pitcher_team) if pitcher_team else None
        logo_url = f"https://www.mlbstatic.com/team-logos/{team_id}.svg" if team_id else None

        top_left, top_right = st.columns([2.2, 1])

        with top_left:
            name_cols = st.columns([0.13, 0.6, 0.13, 1.0])
            with name_cols[0]:
                st.image(headshot_url, width=60)
            with name_cols[1]:
                st.subheader(display_name.upper())
            if logo_url:
                with name_cols[2]:
                    st.image(logo_url, width=45)
            st.caption(f"{start_str} → {end_str}   ·   game_types={','.join(sorted(list(allowed_gt)))}   ·   app={APP_VERSION}")

            st.markdown("### SEASON SUMMARY")
            if season_tbl.empty:
                st.info("No season summary available (likely no FG seasons + no Statcast rows).")
            else:
                season_fmt = {
                    "ERA": "{:.2f}",
                    "FIP": "{:.2f}",
                    "xFIP": "{:.2f}",
                    "K%": "{:.2f}",
                    "BB%": "{:.2f}",
                    "K-BB%": "{:.2f}",
                    "xwOBA": "{:.3f}",
                    "VAA": "{:.1f}",
                    "HAA": "{:.1f}",
                    "vRel": "{:.1f}",
                    "hRel": "{:.1f}",
                }
                st.dataframe(
                    season_tbl.style.format(season_fmt, na_rep="—"),
                    use_container_width=True,
                    hide_index=True,
                )

            st.caption("FanGraphs and Baseball Savant data (xwOBA). Note: If FanGraphs data doesn’t load, enter FG ID — the 5 digits in the player’s FanGraphs page URL.")

        with top_right:
            st.markdown("### QUICK TOTALS")
            c1, c2 = st.columns(2)
            c1.metric("Pitches", f"{len(valid_pitch_rows(sc)):,}")
            c2.metric("Games", f"{sc['game_date'].nunique():,}" if "game_date" in sc.columns else "—")

        st.divider()

        # -----------------------------------------------------
        # PLATOON SPLITS
        # -----------------------------------------------------
        st.markdown("#### Platoon Splits")

        def compute_platoon_splits(sc, hand):
            df = sc[sc["stand"] == hand].copy() if "stand" in sc.columns else pd.DataFrame()
            if df.empty:
                return None
            pa_end = _pa_end_rows(df)
            evs = pa_end["events"].fillna("").astype(str) if not pa_end.empty and "events" in pa_end.columns else pd.Series(dtype=str)
            h = int(evs.isin(["single","double","triple","home_run"]).sum())
            hr = int((evs == "home_run").sum())
            so = int(evs.isin(["strikeout","strikeout_double_play"]).sum())
            bb = int(evs.isin(["walk","intent_walk"]).sum())
            hbp = int((evs == "hit_by_pitch").sum())
            non_ab = {"walk","intent_walk","hit_by_pitch","sac_fly","sac_bunt","catcher_interf"}
            ab = int((~evs.isin(list(non_ab)) & evs.ne("")).sum())
            pa = ab + bb + hbp
            avg = h/ab if ab > 0 else None
            obp = (h+bb+hbp)/pa if pa > 0 else None
            doubles = int((evs == "double").sum())
            triples = int((evs == "triple").sum())
            tb = h - hr - doubles - triples + 2*doubles + 3*triples + 4*hr
            slg = tb/ab if ab > 0 else None
            ops = (obp or 0) + (slg or 0) if obp is not None and slg is not None else None
            k_pct = so/pa*100 if pa > 0 else None
            bb_pct = bb/pa*100 if pa > 0 else None
            ev = float(safe_num(df["launch_speed"]).dropna().mean()) if "launch_speed" in df.columns else None
            xwoba = xwoba_savant_like(df)
            return {
                "Split": f"vs {hand}HH",
                "PA": pa,
                "K%": round(k_pct, 1) if k_pct is not None else np.nan,
                "BB%": round(bb_pct, 1) if bb_pct is not None else np.nan,
                "AVG": round(avg, 3) if avg is not None else np.nan,
                "OBP": round(obp, 3) if obp is not None else np.nan,
                "SLG": round(slg, 3) if slg is not None else np.nan,
                "OPS": round(ops, 3) if ops is not None else np.nan,
                "EV": round(ev, 1) if ev is not None else np.nan,
                "xwOBA": round(xwoba, 3) if xwoba is not None else np.nan,
            }

        splits_L = compute_platoon_splits(sc, "L")
        splits_R = compute_platoon_splits(sc, "R")
        splits_rows = [s for s in [splits_L, splits_R] if s is not None]
        if splits_rows:
            splits_df = pd.DataFrame(splits_rows)
            splits_fmt = {
                "PA": "{:.0f}",
                "K%": "{:.1f}",
                "BB%": "{:.1f}",
                "AVG": "{:.3f}",
                "OBP": "{:.3f}",
                "SLG": "{:.3f}",
                "OPS": "{:.3f}",
                "EV": "{:.1f}",
                "xwOBA": "{:.3f}",
            }
            st.dataframe(splits_df.style.format(splits_fmt, na_rep="—"), use_container_width=True, hide_index=True)

        st.divider()

        # -----------------------------------------------------
        # Movement + Pitch Metrics
        # -----------------------------------------------------
        st.markdown("## PITCH MOVEMENT + METRICS")
        left_plot, right_tbl = st.columns([1.05, 1.55], gap="large")

        with left_plot:
            plot_pitch_break_cloud(sc, compact=True)

        with right_tbl:
            st.markdown("### PITCH METRICS")
            if pitch_metrics_disp.empty:
                st.warning("No pitch metrics available.")
            else:
                fmt_map = {
                    "Pitch%": "{:.1f}",
                    "Pitches": "{:.0f}",
                    "Velo": "{:.1f}",
                    "iVB": "{:.0f}",
                    "HB": "{:.0f}",
                    "Spin": "{:.0f}",
                    "Ext": "{:.2f}",
                    "CalledStr%": "{:.1f}",
                    "SwStr%": "{:.1f}",
                    "Zone%": "{:.1f}",
                    "CSW%": "{:.1f}",
                    "Chase%": "{:.1f}",
                    "ZWhiff%": "{:.1f}",
                    "xwOBA": "{:.3f}",
                    "VAA": "{:.1f}",
                    "HAA": "{:.1f}",
                    "vRel": "{:.1f}",
                    "hRel": "{:.1f}",

                }

                if league_compare and baselines:
                    st.dataframe(
                        style_red_green(
                            pitch_metrics_disp,
                            {
                                "Velo": "high_good",
                                "Spin": "high_good",
                                "Ext": "high_good",
                                "CalledStr%": "high_good",
                                "SwStr%": "high_good",
                                "CSW%": "high_good",
                                "Chase%": "high_good",
                                "ZWhiff%": "high_good",
                                "xwOBA": "low_good",
                            },
                            fmt_map=fmt_map,
                            pitch_col="Pitch",
                            baselines=baselines,
                            baseline_group_col="Pitch",
                            qualify_col="Pitches",
                            qualify_min=MIN_BASELINE_PITCHES,
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )
                    st.caption(f"Pitch shading = z-score vs league baseline for THAT pitch type ({end_date.year} season).")
                else:
                    st.dataframe(
                        pitch_metrics_disp.style.format(fmt_map, na_rep="—"),
                        use_container_width=True,
                        hide_index=True,
                    )

        st.divider()

        # -----------------------------------------------------
        # AGGRESSION + BATTED BALL
        # -----------------------------------------------------
        st.markdown("## AGGRESSION + BATTED BALL")
        z = compute_zone_contact_block(sc)
        lgz = league_zone_contact if (league_compare and isinstance(league_zone_contact, dict)) else {}

        def _delta(val, base):
            if val is None or base is None:
                return None
            try:
                return float(val) - float(base)
            except Exception:
                return None

        r1 = st.columns(5)
        r1[0].metric("Zone%", f"{z['Zone%']:.1f}%" if z["Zone%"] is not None else "—",
                     delta=(f"{_delta(z['Zone%'], lgz.get('Zone%')):+.1f}%"
                            if _delta(z["Zone%"], lgz.get("Zone%")) is not None else None))
        r1[1].metric("First Pitch Strike%", f"{z['First Pitch Strike%']:.1f}%" if z["First Pitch Strike%"] is not None else "—",
                     delta=(f"{_delta(z['First Pitch Strike%'], lgz.get('First Pitch Strike%')):+.1f}%"
                            if _delta(z["First Pitch Strike%"], lgz.get("First Pitch Strike%")) is not None else None))
        r1[2].metric("1-1 Strike%", f"{z['1-1 Strike%']:.1f}%" if z["1-1 Strike%"] is not None else "—",
                     delta=(f"{_delta(z['1-1 Strike%'], lgz.get('1-1 Strike%')):+.1f}%"
                            if _delta(z["1-1 Strike%"], lgz.get("1-1 Strike%")) is not None else None))
        r1[3].metric("AB < 3 Pitches%", f"{z['AB < 3 Pitches%']:.1f}%" if z["AB < 3 Pitches%"] is not None else "—",
                     delta=(f"{_delta(z['AB < 3 Pitches%'], lgz.get('AB < 3 Pitches%')):+.1f}%"
                            if _delta(z["AB < 3 Pitches%"], lgz.get("AB < 3 Pitches%")) is not None else None))
        r1[4].metric("R2K%", f"{z['R2K%']:.1f}%" if z["R2K%"] is not None else "—",
                     delta=(f"{_delta(z['R2K%'], lgz.get('R2K%')):+.1f}%"
                            if _delta(z["R2K%"], lgz.get("R2K%")) is not None else None))

        r2 = st.columns(5)
        r2[0].metric("Swing%", f"{z['Swing%']:.1f}%" if z["Swing%"] is not None else "—",
                     delta=(f"{_delta(z['Swing%'], lgz.get('Swing%')):+.1f}%"
                            if _delta(z["Swing%"], lgz.get("Swing%")) is not None else None))
        r2[1].metric("Exit Velo", f"{z['Exit Velo']:.1f}" if z["Exit Velo"] is not None else "—",
                     delta=(f"{_delta(z['Exit Velo'], lgz.get('Exit Velo')):+.1f}"
                            if _delta(z["Exit Velo"], lgz.get("Exit Velo")) is not None else None))
        r2[2].metric("Launch Angle", f"{z['Launch Angle']:.1f}" if z["Launch Angle"] is not None else "—",
                     delta=(f"{_delta(z['Launch Angle'], lgz.get('Launch Angle')):+.1f}"
                            if _delta(z["Launch Angle"], lgz.get("Launch Angle")) is not None else None))
        r2[3].metric("HardHit%", f"{z['HardHit%']:.1f}%" if z["HardHit%"] is not None else "—",
                     delta=(f"{_delta(z['HardHit%'], lgz.get('HardHit%')):+.1f}%"
                            if _delta(z["HardHit%"], lgz.get("HardHit%")) is not None else None))
        r2[4].metric("BABIP", f"{z['BABIP']:.3f}" if z["BABIP"] is not None else "—",
                     delta=(f"{_delta(z['BABIP'], lgz.get('BABIP')):+.3f}"
                            if _delta(z["BABIP"], lgz.get("BABIP")) is not None else None))

        r3 = st.columns(5)
        r3[0].metric("GB%", f"{z['GB%']:.1f}%" if z["GB%"] is not None else "—",
                     delta=(f"{_delta(z['GB%'], lgz.get('GB%')):+.1f}%"
                            if _delta(z["GB%"], lgz.get("GB%")) is not None else None))
        r3[1].metric("LD%", f"{z['LD%']:.1f}%" if z["LD%"] is not None else "—",
                     delta=(f"{_delta(z['LD%'], lgz.get('LD%')):+.1f}%"
                            if _delta(z["LD%"], lgz.get("LD%")) is not None else None))
        r3[2].metric("FB%", f"{z['FB%']:.1f}%" if z["FB%"] is not None else "—",
                     delta=(f"{_delta(z['FB%'], lgz.get('FB%')):+.1f}%"
                            if _delta(z["FB%"], lgz.get("FB%")) is not None else None))
        r3[3].metric("HR/FB%", f"{z['HR/FB%']:.1f}%" if z["HR/FB%"] is not None else "—",
                     delta=(f"{_delta(z['HR/FB%'], lgz.get('HR/FB%')):+.1f}%"
                            if _delta(z["HR/FB%"], lgz.get("HR/FB%")) is not None else None))
        r3[4].metric("Barrel%", f"{z['Barrel%']:.1f}%" if z["Barrel%"] is not None else "—",
                     delta=(f"{_delta(z['Barrel%'], lgz.get('Barrel%')):+.1f}%"
                            if _delta(z["Barrel%"], lgz.get("Barrel%")) is not None else None))

        st.caption("AB < 3 Pitches% = share of plate appearances that end in 1–2 pitches. R2K% = % of AB where count reaches 0-2.")
        st.divider()

        # -----------------------------------------------------
        # HEATMAPS
        # -----------------------------------------------------
        st.markdown("## HEATMAPS")
        st.caption("Fastballs / Offspeed / Breaking for LHH and RHH. Red = more frequent / harder contact; Blue = less.")
        hL, hR = st.columns(2, gap="large")

        with hL:
            st.markdown("### vs LHH")
            for grp in PITCH_GROUP_ORDER:
                plot_heatmap_contour(sc, "L", grp, heat_mode)

        with hR:
            st.markdown("### vs RHH")
            for grp in PITCH_GROUP_ORDER:
                plot_heatmap_contour(sc, "R", grp, heat_mode)

        # -----------------------------------------------------
        st.divider()

        # -----------------------------------------------------
        # USAGE
        # -----------------------------------------------------
        st.markdown("## USAGE")
        uL, uR = st.columns(2, gap="large")

        with uL:
            st.markdown("### vs LHH")
            tbl = build_usage_situation_table(sc, "L")
            if tbl.empty:
                st.info("No LHH data.")
            else:
                value_cols = [c for c in tbl.columns if c != "Pitch"]
                st.dataframe(style_usage_delta_table(tbl, value_cols=value_cols), use_container_width=True, hide_index=True)
                st.caption("Shading is relative to All Counts: greener = used more, redder = used less.")

        with uR:
            st.markdown("### vs RHH")
            tbl = build_usage_situation_table(sc, "R")
            if tbl.empty:
                st.info("No RHH data.")
            else:
                value_cols = [c for c in tbl.columns if c != "Pitch"]
                st.dataframe(style_usage_delta_table(tbl, value_cols=value_cols), use_container_width=True, hide_index=True)
                st.caption("Shading is relative to All Counts: greener = used more, redder = used less.")

        st.divider()

        # -----------------------------------------------------
        # TRENDS
        # -----------------------------------------------------
        st.markdown("## TRENDS")

        pitch_types = sorted(valid_pitch_rows(sc)["pitch_type"].dropna().astype(str).unique().tolist()) if "pitch_type" in sc.columns else []
        trend_keys = list(TREND_LABELS.keys())
        all_trend_options = trend_keys + list(usage_cols.keys())
        all_trend_labels = {**TREND_LABELS, **usage_cols}

        tc1, tc2 = st.columns(2)
        with tc1:
            trend_pitch = st.selectbox(
                "Pitch filter (does NOT refetch data)",
                options=["(All)"] + pitch_types,
                index=0,
                key="trend_pitch_filter_live",
            )
        with tc2:
            trend_var_key = st.selectbox(
                "Metric",
                options=all_trend_options,
                index=0,
                format_func=lambda k: all_trend_labels.get(k, k),
                key="trend_metric_live",
            )
        trend_vars = [trend_var_key]

        if not trend_vars:
            st.info("Select a trend metric above.")
            return

        # Handle pitch_usage specially - always show all pitches as colored lines
        if "pitch_usage" in trend_vars:
            if "pitch_type" in sc.columns and "game_pk" in sc.columns and "game_date" in sc.columns:
                valid_sc = valid_pitch_rows(sc)
                pitch_types_present = sorted(valid_sc["pitch_type"].dropna().astype(str).unique().tolist())
                game_totals = valid_sc.groupby(["game_pk","game_date"])["pitch_type"].count().rename("total").reset_index()
                fig = go.Figure()
                for ptype in pitch_types_present:
                    pt_counts = valid_sc[valid_sc["pitch_type"]==ptype].groupby(["game_pk","game_date"])["pitch_type"].count().rename("count").reset_index()
                    merged = pt_counts.merge(game_totals, on=["game_pk","game_date"])
                    merged["usage"] = merged["count"] / merged["total"] * 100
                    merged = merged.sort_values("game_date")
                    color = PITCH_COLORS.get(ptype, "#9e9e9e")
                    label = PITCH_NAMES.get(ptype, ptype)
                    hover = "Date: " + pd.to_datetime(merged["game_date"]).dt.strftime("%Y-%m-%d").astype(str) + "<br>" + label + ": " + merged["usage"].round(1).astype(str) + "%"
                    fig.add_trace(go.Scatter(
                        x=pd.to_datetime(merged["game_date"]),
                        y=merged["usage"],
                        mode="lines+markers",
                        name=label,
                        line=dict(color=color),
                        hovertext=hover,
                        hoverinfo="text",
                    ))
                fig.update_layout(
                    height=420,
                    margin=dict(l=40, r=20, t=40, b=40),
                    xaxis=dict(title="Date", tickformat="%b %d", showgrid=True, tickangle=-25),
                    yaxis=dict(title="Usage%", showgrid=True),
                    legend=dict(orientation="v", x=1.02, y=1.0),
                    hovermode="x unified",
                )
                st.plotly_chart(fig, use_container_width=True)
            remaining = [v for v in trend_vars if v != "pitch_usage"]
            if not remaining:
                return
            trend_vars = remaining

        if trend_pitch == "(All)" and "pitch_type" in sc.columns:
            pitch_types_present = sorted(valid_pitch_rows(sc)["pitch_type"].dropna().astype(str).unique().tolist())
            fig = go.Figure()
            opp = None
            for ptype in pitch_types_present:
                tr_p = trend_by_game(sc, trend_vars, pitch_filter=ptype)
                if tr_p.empty:
                    continue
                color = PITCH_COLORS.get(ptype, "#9e9e9e")
                label = PITCH_NAMES.get(ptype, ptype)
                for v in trend_vars:
                    if v not in tr_p.columns:
                        continue
                    y = safe_num(tr_p[v])
                    hover = "Date: " + tr_p["game_date"].dt.strftime("%Y-%m-%d").astype(str) + "<br>" + label + ": " + y.round(3).astype(str)
                    fig.add_trace(go.Scatter(
                        x=tr_p["game_date"],
                        y=y,
                        mode="lines+markers",
                        name=f"{label}" + (f" ({TREND_LABELS.get(v,v)})" if len(trend_vars) > 1 else ""),
                        line=dict(color=color),
                        hovertext=hover,
                        hoverinfo="text",
                    ))
            fig.update_layout(
                height=420,
                margin=dict(l=40, r=20, t=40, b=40),
                xaxis=dict(title="Date", tickformat="%b %d", showgrid=True, tickangle=-25),
                yaxis=dict(title=all_trend_labels.get(trend_vars[0], trend_vars[0]) if trend_vars else "Value", showgrid=True),
                legend=dict(orientation="v", x=1.02, y=1.0),
                hovermode="x unified",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            tr = trend_by_game(sc, trend_vars, pitch_filter=trend_pitch)
            if tr.empty:
                st.info("No trend rows for this selection (try a wider range or different pitch filter).")
                return
            plot_trends_plotly(tr, trend_vars, normalize=normalize)


if __name__ == "__main__":
    main()