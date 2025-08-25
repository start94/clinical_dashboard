from __future__ import annotations
import pandas as pd
import numpy as np

def total_minutes_per_visit(df: pd.DataFrame) -> pd.Series:
    if "visit_id" not in df.columns:
        return pd.Series(dtype=float)
    return df.groupby("visit_id")["minutes"].sum()

def avg_minutes_per_visit(df: pd.DataFrame) -> float:
    if "visit_id" not in df.columns:
        return 0.0
    # handle empty series
    total_minutes = total_minutes_per_visit(df)
    if total_minutes.empty:
        return 0.0
    return float(total_minutes.mean())

def share_time_by_activity(df: pd.DataFrame) -> pd.DataFrame:
    if "minutes" not in df.columns or "activity" not in df.columns:
        return pd.DataFrame(columns=["minutes", "percent"])
    tot = df["minutes"].sum()
    if tot == 0:
        return pd.DataFrame(columns=["minutes", "percent"])
    by_act = df.groupby("activity")["minutes"].sum().sort_values(ascending=False)
    pct = (by_act / tot * 100).round(1)
    return pd.DataFrame({"minutes": by_act, "percent": pct})

def avg_after_hours_minutes_per_visit(df: pd.DataFrame) -> float:
    if "visit_id" not in df.columns or "is_after_hours" not in df.columns or "minutes" not in df.columns:
        return 0.0
    def _after(x):
        return x.loc[x["is_after_hours"], "minutes"].sum()
    # handle empty series
    per_visit = df.groupby("visit_id").apply(_after)
    if per_visit.empty:
        return 0.0
    return float(per_visit.mean())

def ai_note_share(df: pd.DataFrame) -> float:
    if "visit_id" not in df.columns or "is_ai_note" not in df.columns:
        return 0.0
    # quota visite con almeno una nota documentale AI
    per_visit_ai = df.groupby("visit_id")["is_ai_note"].max()
    # handle empty series
    if per_visit_ai.empty:
        return 0.0
    return float(per_visit_ai.mean() * 100)

def ai_correction_avg_minutes(df: pd.DataFrame) -> float:
    if "activity" not in df.columns or "is_ai_note" not in df.columns or "ai_edit_minutes" not in df.columns:
        return 0.0
    ai_docs = df[(df["activity"] == "documentation") & (df["is_ai_note"])]
    if ai_docs.empty:
        return 0.0
    return float(ai_docs["ai_edit_minutes"].mean())

def clinicians_workload(df: pd.DataFrame) -> pd.DataFrame:
    if "clinician_id" not in df.columns or "minutes" not in df.columns:
        return pd.DataFrame(columns=["clinician_id", "total_minutes"])
    per_clin = df.groupby("clinician_id")["minutes"].sum().sort_values(ascending=False).reset_index()
    per_clin.rename(columns={"minutes": "total_minutes"}, inplace=True)
    return per_clin

def outlier_visits(df: pd.DataFrame) -> pd.DataFrame:
    if "visit_id" not in df.columns:
        return pd.DataFrame(columns=["visit_id", "total_minutes"])
    tv = total_minutes_per_visit(df).reset_index(name="total_minutes")
    if tv.empty:
        return pd.DataFrame(columns=["visit_id", "total_minutes"])
    q1, q3 = tv["total_minutes"].quantile([0.25, 0.75])
    iqr = q3 - q1
    cut = q3 + 1.5 * iqr
    return tv[tv["total_minutes"] > cut].sort_values("total_minutes", ascending=False)

def kpi_overview(df: pd.DataFrame) -> dict:
    return {
        "avg_minutes_per_visit": round(avg_minutes_per_visit(df), 1),
        "avg_after_hours_minutes_per_visit": round(avg_after_hours_minutes_per_visit(df), 2),
        "ai_note_share_percent": round(ai_note_share(df), 1),
        "ai_correction_avg_minutes": round(ai_correction_avg_minutes(df), 2),
    }