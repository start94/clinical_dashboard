from __future__ import annotations
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil import tz
import fitz  # PyMuPDF

ACTIVITIES = ["documentation", "chart_review", "orders", "inbox"]

def create_synthetic_logs(n_visits: int = 300, n_clinicians: int = 10, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    clinicians = [f"C{idx:02d}" for idx in range(1, n_clinicians + 1)]
    departments = ["Cardiologia", "Pronto Soccorso", "Medicina Generale / Interna", "Neurologia", "Chirurgia Generale", "Pediatria", "Ortopedia e Traumatologia"]  # internal med, cardiology, etc.

    rows = []
    base_day = datetime.now().replace(hour=9, minute=0, second=0, microsecond=0)
    visit_id_counter = 0
    visits_per_department = n_visits // len(departments)

    for dept in departments:
        for _ in range(visits_per_department):
            visit_id = f"V{{visit_id_counter:05d}}"
            clinician = rng.choice(clinicians)
            # distribuzione dei minuti per attività (sommatoria ≈ 12–20 min/visita)
            doc = max(4, int(rng.normal(6, 2)))
            rev = max(3, int(rng.normal(5, 2)))
            ords = max(2, int(rng.normal(4, 1)))
            inbox = max(1, int(rng.normal(2, 1)))
            buckets = [doc, rev, ords, inbox]
            start = base_day + timedelta(minutes=int(rng.integers(0, 60 * 6)))  # in fascia 9-15
            t = start

            # 20–35% visite con nota AI
            ai_flag = rng.random() < rng.uniform(0.2, 0.35)
            # tempo correzione AI se presente
            ai_edit = int(max(0, rng.normal(1.5, 0.8))) if ai_flag else 0

            for act, mins in zip(ACTIVITIES, buckets):
                end = t + timedelta(minutes=mins)
                is_after = end.hour >= 18
                rows.append({
                    "visit_id": visit_id,
                    "clinician_id": clinician,
                    "department": dept,
                    "activity": act,
                    "start_time": t,
                    "end_time": end,
                    "minutes": mins,
                    "is_after_hours": bool(is_after),
                    "is_ai_note": bool(ai_flag if act == "documentation" else False),
                    "ai_edit_minutes": ai_edit if act == "documentation" and ai_flag else 0,
                })
                t = end

            # ~10% di lavoro extra-orario (pajama time)
            if rng.random() < 0.1:
                extra = int(max(5, rng.normal(12, 5)))
                rows.append({
                    "visit_id": visit_id,
                    "clinician_id": clinician,
                    "department": dept,
                    "activity": "documentation",
                    "start_time": t.replace(hour=19, minute=int(rng.integers(0, 59))),
                    "end_time": t.replace(hour=19, minute=int(rng.integers(0, 59))) + timedelta(minutes=extra),
                    "minutes": extra,
                    "is_after_hours": True,
                    "is_ai_note": False,
                    "ai_edit_minutes": 0,
                })
            visit_id_counter += 1

    df = pd.DataFrame(rows)
    return df

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["start_time", "end_time"])
    # se manca la colonna minutes, calcolala
    if "minutes" not in df.columns:
        df["minutes"] = (df["end_time"] - df["start_time"]).dt.total_seconds() // 60
    # cast booleani se arrivano come 0/1
    for col in ["is_after_hours", "is_ai_note"]:
        if col in df.columns:
            df[col] = df[col].astype(bool)
    return df

def load_pdf(file) -> pd.DataFrame:
    """
    Estrae tabelle da un file PDF e le converte in un DataFrame pandas.
    """
    pdf_document = fitz.open(stream=file.read(), filetype="pdf")
    all_tables = []

    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        tables = page.find_tables()
        if tables:
            for table in tables:
                all_tables.append(table.to_pandas())

    if not all_tables:
        return pd.DataFrame() # Ritorna un DF vuoto se non ci sono tabelle

    # Concatena tutti i DataFrame delle tabelle trovate
    df = pd.concat(all_tables, ignore_index=True)

    # --- Post-processing simile a load_csv ---
    # Converte le colonne di data/ora se presenti
    for col in ["start_time", "end_time"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # Calcola i minuti se necessario
    if "minutes" not in df.columns and "start_time" in df.columns and "end_time" in df.columns:
        df["minutes"] = (df["end_time"] - df["start_time"]).dt.total_seconds() // 60

    # Converte colonne booleane
    for col in ["is_after_hours", "is_ai_note"]:
        if col in df.columns:
            # Converte 0/1, "True"/"False" in booleani
            if df[col].dtype != bool:
                df[col] = df[col].apply(lambda x: str(x).lower() in ['true', '1', 'yes', 'y'])

    return df

def extract_text_from_pdf(file) -> str:
    """
    Extracts text from a PDF file.
    """
    pdf_document = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    return text
