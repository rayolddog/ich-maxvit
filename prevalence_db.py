#!/usr/bin/env python3
"""
Prevalence database for the ICH AI pipeline.

Stores one row per processed study in a local SQLite database.
Supports querying prevalence by institutional location and time window,
enabling the ICH agent to substitute locally observed prevalence for the
generic indication-based defaults.

Schema
------
  study_results
    study_uid           TEXT  PRIMARY KEY
    patient_id          TEXT
    exam_date           TEXT  ISO date (YYYY-MM-DD)    from DICOM StudyDate
    exam_datetime       TEXT  ISO datetime             from DICOM StudyDate+Time
    location            TEXT  canonical unit label
    raw_department      TEXT  InstitutionalDepartmentName as read from DICOM
    ai_positive         INTEGER  0/1
    dominant_class      TEXT
    prob_any            REAL
    prob_epidural       REAL
    prob_intraparenchymal REAL
    prob_intraventricular REAL
    prob_subarachnoid   REAL
    prob_subdural       REAL
    series_folder       TEXT
    checkpoint          TEXT
    processed_datetime  TEXT  ISO datetime  when the AI ran

Usage
-----
    from prevalence_db import PrevalenceDB
    db = PrevalenceDB()
    db.record_study(study_uid=..., location="ER", ai_positive=True, ...)
    prev, n, n_pos = db.get_prevalence(location="ER", days=365)
    print(db.summary_table(days=365))
"""

import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

DEFAULT_DB_PATH = Path(__file__).parent / "checkpoints_maxvit" / "prevalence.db"

# Maps raw DICOM InstitutionalDepartmentName values (lowercased, stripped)
# to canonical location labels.  Extend to match local site conventions.
LOCATION_MAP: dict[str, str] = {
    "emergency":          "ER",
    "ed":                 "ER",
    "er":                 "ER",
    "emergency department": "ER",
    "emergency medicine": "ER",
    "trauma":             "ER",
    "neurology":          "Neurology",
    "neuro":              "Neurology",
    "neuroscience":       "Neurology",
    "neurosurgery":       "Neurosurgery",
    "neuro icu":          "Neuro ICU",
    "neurological icu":   "Neuro ICU",
    "nicu":               "Neuro ICU",
    "nsicu":              "Neuro ICU",
    "icu":                "ICU",
    "intensive care":     "ICU",
    "micu":               "ICU",
    "sicu":               "ICU",
    "medical":            "Med/Surg",
    "medicine":           "Med/Surg",
    "internal medicine":  "Med/Surg",
    "med/surg":           "Med/Surg",
    "surgical":           "Med/Surg",
    "outpatient":         "Outpatient",
    "ambulatory":         "Outpatient",
    "clinic":             "Outpatient",
    "radiology":          "Radiology",
    "inpatient":          "Inpatient",
}


def normalize_location(raw: str) -> str:
    """Map raw DICOM department name to a canonical location label."""
    if not raw:
        return "Unknown"
    key = raw.strip().lower()
    # Exact match first
    if key in LOCATION_MAP:
        return LOCATION_MAP[key]
    # Substring match
    for fragment, label in LOCATION_MAP.items():
        if fragment in key:
            return label
    # Return title-cased raw value as fallback
    return raw.strip().title()


class PrevalenceDB:
    """
    Thread-safe SQLite database for AI ICH prevalence tracking.

    Parameters
    ----------
    db_path : str or Path
        Path to the SQLite database file.  Created on first use.
    """

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_schema()

    @contextmanager
    def _conn(self):
        """Per-thread connection with WAL mode for concurrent reads."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        yield self._local.conn

    def _init_schema(self):
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS study_results (
                    study_uid             TEXT PRIMARY KEY,
                    patient_id            TEXT,
                    exam_date             TEXT,
                    exam_datetime         TEXT,
                    location              TEXT,
                    raw_department        TEXT,
                    ai_positive           INTEGER,
                    dominant_class        TEXT,
                    prob_any              REAL,
                    prob_epidural         REAL,
                    prob_intraparenchymal REAL,
                    prob_intraventricular REAL,
                    prob_subarachnoid     REAL,
                    prob_subdural         REAL,
                    series_folder         TEXT,
                    checkpoint            TEXT,
                    processed_datetime    TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_location_date
                ON study_results (location, exam_date)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_exam_date
                ON study_results (exam_date)
            """)
            conn.commit()

    # ── Write ──────────────────────────────────────────────────────────────────

    def record_study(
        self,
        study_uid:             str,
        ai_positive:           bool,
        patient_id:            str  = "",
        exam_date:             str  = "",
        exam_datetime:         str  = "",
        location:              str  = "",
        raw_department:        str  = "",
        dominant_class:        str  = "",
        study_level_probs:     dict = None,
        series_folder:         str  = "",
        checkpoint:            str  = "",
    ) -> bool:
        """
        Insert or replace a study result.  Returns True if this is a new study.

        Parameters
        ----------
        study_level_probs : dict  {class_name: {"prob": float, ...}}
            As returned by ich_inference.run_inference()["study_level"].
        """
        probs = study_level_probs or {}

        def p(cls):
            return float(probs.get(cls, {}).get("prob", 0.0))

        now = datetime.now().isoformat()
        if not exam_date:
            exam_date = now[:10]
        if not exam_datetime:
            exam_datetime = now

        canonical = normalize_location(raw_department or location)

        with self._conn() as conn:
            existing = conn.execute(
                "SELECT 1 FROM study_results WHERE study_uid = ?", (study_uid,)
            ).fetchone()

            conn.execute("""
                INSERT OR REPLACE INTO study_results (
                    study_uid, patient_id, exam_date, exam_datetime,
                    location, raw_department,
                    ai_positive, dominant_class,
                    prob_any, prob_epidural, prob_intraparenchymal,
                    prob_intraventricular, prob_subarachnoid, prob_subdural,
                    series_folder, checkpoint, processed_datetime
                ) VALUES (
                    ?, ?, ?, ?,
                    ?, ?,
                    ?, ?,
                    ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?
                )
            """, (
                study_uid, patient_id, exam_date, exam_datetime,
                canonical, raw_department,
                1 if ai_positive else 0, dominant_class,
                p("any"), p("epidural"), p("intraparenchymal"),
                p("intraventricular"), p("subarachnoid"), p("subdural"),
                series_folder, checkpoint, now,
            ))
            conn.commit()
            return existing is None

    # ── Read ───────────────────────────────────────────────────────────────────

    def get_prevalence(
        self,
        location: Optional[str] = None,
        days:     Optional[int] = None,
        min_n:    int = 10,
    ) -> tuple[float, int, int]:
        """
        Return (prevalence, n_total, n_positive) for the given filters.

        prevalence is None if fewer than min_n studies meet the criteria.

        Parameters
        ----------
        location : str or None
            Canonical location label (e.g. "ER").  None = all locations.
        days : int or None
            Restrict to studies with exam_date >= today - days.
        min_n : int
            Minimum study count; returns (None, n, n_pos) if below threshold.
        """
        where, params = self._build_where(location, days)
        with self._conn() as conn:
            row = conn.execute(
                f"SELECT COUNT(*) AS n, SUM(ai_positive) AS n_pos "
                f"FROM study_results {where}",
                params,
            ).fetchone()
        n     = row["n"] or 0
        n_pos = int(row["n_pos"] or 0)
        if n < min_n:
            return (None, n, n_pos)
        return (round(n_pos / n, 6), n, n_pos)

    def summary_table(
        self,
        days: Optional[int] = None,
        min_n: int = 10,
    ) -> list[dict]:
        """
        Return prevalence for every location in the DB.

        Each row: {location, n_total, n_positive, prevalence, period_days}
        Rows with fewer than min_n studies get prevalence=None.
        Sorted by n_total descending.
        """
        where, params = self._build_where(None, days)
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT location, COUNT(*) AS n, SUM(ai_positive) AS n_pos "
                f"FROM study_results {where} "
                f"GROUP BY location ORDER BY n DESC",
                params,
            ).fetchall()
        result = []
        for row in rows:
            n     = row["n"]
            n_pos = int(row["n_pos"] or 0)
            prev  = round(n_pos / n, 6) if n >= min_n else None
            result.append({
                "location":    row["location"],
                "n_total":     n,
                "n_positive":  n_pos,
                "prevalence":  prev,
                "period_days": days,
            })
        return result

    def trend_report(
        self,
        location:   Optional[str] = None,
        period:     str = "month",
        n_periods:  int = 12,
        min_n:      int = 10,
    ) -> list[dict]:
        """
        Prevalence broken into consecutive time buckets.

        Parameters
        ----------
        period : "week" | "month" | "year"
        n_periods : how many buckets to return (most recent first)

        Returns list of dicts: {period_label, start_date, end_date,
                                n_total, n_positive, prevalence}
        """
        today = datetime.now().date()
        buckets = []

        for i in range(n_periods):
            if period == "week":
                end   = today - timedelta(weeks=i)
                start = end   - timedelta(weeks=1)
                label = f"W{end.isocalendar()[1]:02d} {end.year}"
            elif period == "year":
                end   = today.replace(year=today.year - i, month=12, day=31)
                start = end.replace(month=1, day=1)
                label = str(end.year)
            else:  # month
                # Walk back i months
                month = (today.month - 1 - i) % 12 + 1
                year  = today.year + ((today.month - 1 - i) // 12)
                import calendar
                last_day = calendar.monthrange(year, month)[1]
                start = datetime(year, month, 1).date()
                end   = datetime(year, month, last_day).date()
                label = f"{year}-{month:02d}"

            where_parts = ["exam_date >= ?", "exam_date <= ?"]
            params      = [str(start), str(end)]
            if location:
                where_parts.append("location = ?")
                params.append(location)
            where = "WHERE " + " AND ".join(where_parts)

            with self._conn() as conn:
                row = conn.execute(
                    f"SELECT COUNT(*) AS n, SUM(ai_positive) AS n_pos "
                    f"FROM study_results {where}",
                    params,
                ).fetchone()
            n     = row["n"] or 0
            n_pos = int(row["n_pos"] or 0)
            buckets.append({
                "period_label": label,
                "start_date":   str(start),
                "end_date":     str(end),
                "n_total":      n,
                "n_positive":   n_pos,
                "prevalence":   round(n_pos / n, 6) if n >= min_n else None,
            })

        return buckets  # most recent first

    def best_prevalence_for_agent(
        self,
        location:   Optional[str] = None,
        days:       int = 365,
        min_n:      int = 30,
        fallback:   Optional[float] = None,
    ) -> tuple[float, str]:
        """
        Return (prevalence, source_description) for use by the ICH agent.

        Tries progressively broader queries if the narrow one lacks data:
          1. location + days window
          2. location + all time
          3. all locations + days window
          4. fallback value

        Parameters
        ----------
        fallback : float or None
            Value to use if no DB query meets min_n.  If None, returns None.
        """
        attempts = [
            (location, days,  f"local {location}, last {days}d"),
            (location, None,  f"local {location}, all time"),
            (None,     days,  f"all locations, last {days}d"),
            (None,     None,  "all locations, all time"),
        ]
        for loc, d, desc in attempts:
            if loc is None and location is not None and d == days:
                continue  # skip redundant all-location narrow before trying wide
            prev, n, n_pos = self.get_prevalence(loc, d, min_n)
            if prev is not None:
                return (prev, f"observed ({desc}, n={n}, {n_pos} positive)")

        if fallback is not None:
            return (fallback, "indication default (insufficient local data)")
        return (None, "no prevalence data available")

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _build_where(self, location, days):
        parts, params = [], []
        if location:
            parts.append("location = ?")
            params.append(location)
        if days:
            cutoff = (datetime.now() - timedelta(days=days)).date().isoformat()
            parts.append("exam_date >= ?")
            params.append(cutoff)
        where = ("WHERE " + " AND ".join(parts)) if parts else ""
        return where, params

    def print_summary(self, days: Optional[int] = None, min_n: int = 10):
        """Print a formatted prevalence summary table to stdout."""
        rows   = self.summary_table(days, min_n)
        period = f"last {days}d" if days else "all time"
        total_prev, total_n, total_pos = self.get_prevalence(None, days, min_n=1)

        W = 68
        print(f"\n{'='*W}")
        print(f"  AI ICH Prevalence Summary — {period}")
        print(f"  Database: {self.db_path}")
        print(f"{'='*W}")
        print(f"  {'Location':<22} {'Studies':>8} {'ICH+':>6} {'Prevalence':>12}")
        print(f"  {'-'*62}")
        for row in rows:
            prev_str = (f"{row['prevalence']*100:.2f}%"
                        if row["prevalence"] is not None
                        else f"<{min_n} studies")
            print(f"  {row['location']:<22} {row['n_total']:>8,} "
                  f"{row['n_positive']:>6,} {prev_str:>12}")
        print(f"  {'-'*62}")
        prev_all = f"{total_prev*100:.2f}%" if total_prev is not None else "—"
        print(f"  {'TOTAL':<22} {total_n:>8,} {total_pos:>6,} {prev_all:>12}")
        print(f"{'='*W}\n")

    def print_trend(
        self,
        location:  Optional[str] = None,
        period:    str = "month",
        n_periods: int = 12,
        min_n:     int = 10,
    ):
        """Print a time-series prevalence table to stdout."""
        rows  = self.trend_report(location, period, n_periods, min_n)
        title = f"{location or 'All locations'} — {period}ly trend"
        W     = 60
        print(f"\n{'='*W}")
        print(f"  AI ICH Prevalence Trend: {title}")
        print(f"{'='*W}")
        print(f"  {'Period':<12} {'Studies':>8} {'ICH+':>6} {'Prevalence':>12}")
        print(f"  {'-'*50}")
        for row in reversed(rows):  # chronological order
            prev_str = (f"{row['prevalence']*100:.2f}%"
                        if row["prevalence"] is not None
                        else f"<{min_n}")
            print(f"  {row['period_label']:<12} {row['n_total']:>8,} "
                  f"{row['n_positive']:>6,} {prev_str:>12}")
        print(f"{'='*W}\n")
