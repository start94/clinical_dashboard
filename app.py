import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pymongo import MongoClient
import os
import json
from dotenv import load_dotenv

# --- Import moduli locali (per dashboard burocrazia) ---
from src.utils import create_synthetic_logs, load_csv, load_pdf
from src.kpi import (
    share_time_by_activity,
    clinicians_workload,
    outlier_visits,
    kpi_overview,
)
from src.prediction import (
    load_and_preprocess_data,
    train_and_save_model,
    load_model_and_predict,
)

# --- Impostazioni generali ---
st.set_page_config(page_title="Healthcare Dashboard", layout="wide")

# --- Dati Reparti ---
REPARTI = {
    "🫀 Medicina Interna e Specialità Mediche": [
        "Medicina Generale / Interna", "Cardiologia", "Pneumologia", "Gastroenterologia",
        "Endocrinologia", "Nefrologia", "Reumatologia", "Ematologia", "Malattie Infettive",
        "Allergologia e Immunologia", "Geriatria"
    ],
    "🧠 Neurologia e Psichiatria": [
        "Neurologia", "Neurofisiopatologia", "Psichiatria", "Neuropsichiatria Infantile"
    ],
    "🦴 Chirurgia e Specialità Chirurgiche": [
        "Chirurgia Generale", "Chirurgia Vascolare", "Chirurgia Toracica",
        "Chirurgia Plastica e Ricostruttiva", "Chirurgia Maxillo-Facciale",
        "Ortopedia e Traumatologia", "Neurochirurgia", "Urologia", "Proctologia"
    ],
    "👶 Ostetricia, Ginecologia e Pediatria": [
        "Ostetricia e Ginecologia", "Sala Parto", "Neonatologia",
        "Terapia Intensiva Neonatale (TIN)", "Pediatria", "Pediatria Specialistica"
    ],
    "👁️‍🗨️ Specialità Sensoriali e Dermatologiche": [
        "Oculistica (Oftalmologia)", "Otorinolaringoiatria (ORL)", "Dermatologia"
    ],
    "🧬 Oncologia e Terapie": [
        "Oncologia Medica", "Radioterapia", "Medicina Nucleare", "Terapia del Dolore",
        "Cure Palliative"
    ],
    "🧪 Diagnostica e Laboratori": [
        "Radiologia / Diagnostica per Immagini", "Laboratorio Analisi",
        "Anatomia Patologica", "Medicina di Laboratorio"
    ],
    "🚑 Emergenza e Terapie Intensive": [
        "Pronto Soccorso", "Medicina d’Urgenza", "Terapia Intensiva", "Rianimazione",
        "Unità Coronarica (UTIC)", "Stroke Unit"
    ],
    "🧘‍♂️ Riabilitazione e Servizi di Supporto": [
        "Medicina Fisica e Riabilitativa", "Fisioterapia", "Logopedia",
        "Nutrizione Clinica", "Psicologia Clinica", "Servizi Sociali Ospedalieri"
    ]
}

# Carica variabili ambiente
load_dotenv()

# Connessione MongoDB (se configurato)
MONGO_URI = os.getenv("PYMONGO_KEY")
client = None
pazienti_collection = None
ricoveri_simulati_collection = None
if MONGO_URI:
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        client.server_info() # Forza la connessione per verificare che sia attiva
        db = client["cartelle_cliniche"]
        pazienti_collection = db["pazienti"]
        ricoveri_simulati_collection = db["ricoveri_simulati"]
    except Exception as e:
        st.sidebar.error(f"Errore connessione MongoDB: {e}")
        client = None
        pazienti_collection = None
        ricoveri_simulati_collection = None

# --- Sidebar: scelta dataset ---
st.sidebar.header("Sorgente Dati")
dataset_type = st.sidebar.radio("Origine", ["Ricoveri Clinici", "Burocrazia EHR", "Carica CSV Burocrazia"])


# =====================================================================
# 1 SEZIONE MONGODB / JSON (Cartelle Cliniche & Ricoveri)
# =====================================================================
if dataset_type == "Ricoveri Clinici":
    st.title("🏥 Analisi Ricoveri Ospedalieri")

    docs = None
    # Prova a caricare da MongoDB se configurato e connesso
    if pazienti_collection is not None:
        docs = list(pazienti_collection.find())
        st.sidebar.success("Connesso a MongoDB")
# aggiungere anche la collezione ricoveri_simulati se serve
    elif ricoveri_simulati_collection is not None:
        docs = list(ricoveri_simulati_collection.find())
        st.sidebar.success("Connesso a MongoDB")    
        
    # Se MongoDB non è disponibile o fallisce, carica il file JSON locale
    if docs is None:
        st.sidebar.warning("MongoDB non disponibile. Carico 'pazienti.json'...")
        try:
            with open("pazienti.json", "r", encoding="utf-8") as f:
                docs = json.load(f)
            st.sidebar.info("Dati caricati da pazienti.json")
        except FileNotFoundError:
            st.error("❌ 'pazienti.json' non trovato e connessione a MongoDB non riuscita.")
            st.stop()
        except json.JSONDecodeError:
            st.error("❌ Errore di formato in 'pazienti.json'. Controlla il file.")
            st.stop()

    # Controlla se ci sono dati da visualizzare
    if not docs:
        st.warning("⚠️ Nessun dato da visualizzare. Controlla MongoDB o il file 'pazienti.json'.")
        st.stop()
    
    # Normalizza dati
    records = []
    for p in docs:
        nome = p.get("nome", "N/A")
        for r in p.get("ricoveri", []):
            # Conversione sicura delle date con gestione dei valori mancanti
            try:
                data_ricovero = pd.to_datetime(r.get("data_ricovero"))
                data_dimissione = pd.to_datetime(r.get("data_dimissione"))
                giorni = (data_dimissione - data_ricovero).days
            except (TypeError, ValueError):
                data_ricovero, data_dimissione, giorni = None, None, None

            records.append({
                "nome": nome,
                "diagnosi": r.get("diagnosi"),
                "reparto": r.get("reparto", "N/A"),
                "data_ricovero": data_ricovero,
                "data_dimissione": data_dimissione,
                "giorni_ricovero": giorni,
            })

    df = pd.DataFrame(records)

    # --- KPI ---
    media_giorni = df["giorni_ricovero"].mean()
    totale_pazienti = df["nome"].nunique()
    totale_ricoveri = len(df)

    c1, c2, c3 = st.columns(3)
    c1.metric("🧑‍⚕️ Pazienti unici", totale_pazienti)
    c2.metric("🏥 Ricoveri totali", totale_ricoveri)
    c3.metric("⏱ Media giorni ricovero", f"{media_giorni:.1f}")

    st.divider()

    # --- Tabella ---
    st.subheader("📋 Dettagli ricoveri")
    st.dataframe(df.dropna(subset=["data_ricovero", "data_dimissione"]))

    # --- Grafico ---
    st.subheader("📈 Distribuzione durata ricoveri")
    st.bar_chart(df["giorni_ricovero"].dropna())

    # --- Download ---
    st.download_button(
        "⬇️ Scarica ricoveri (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="ricoveri.csv",
        mime="text/csv",
    )

    # --- Sezione Previsione ---
    st.sidebar.title("🔮 Previsione Giorni di Ricovero")
    model_path = "modello_dimissione.joblib"
    if not os.path.exists(model_path):
        with st.spinner("Addestramento del modello in corso..."):
            train_df = load_and_preprocess_data("simulated_ricoveri.json")  # prende i dati da un file JSON
            train_and_save_model(train_df, model_path=model_path)
            st.sidebar.success("Modello addestrato e salvato!")
    
    diagnosi_list = sorted(df["diagnosi"].dropna().unique())
    reparto_list = sorted(df["reparto"].dropna().unique())

    diagnosi_input = st.sidebar.selectbox("Diagnosi", diagnosi_list)
    reparto_input = st.sidebar.selectbox("Reparto", reparto_list)

    if st.sidebar.button("Previeni Durata Ricovero"):
        input_data = pd.DataFrame({
            'diagnosi': [diagnosi_input],
            'reparto': [reparto_input]
        })
        prediction = load_model_and_predict(input_data, model_path=model_path)
        if prediction is not None:
            st.sidebar.metric("Giorni di ricovero previsti", f"{prediction:.1f} giorni")
        else:
            st.sidebar.error("Errore nel caricare il modello.")


# =====================================================================
# 2) SEZIONE CSV e pdf / SINTETICI (Burocrazia clinica)
# =====================================================================
elif dataset_type == "Burocrazia EHR":
    st.title("🩺 Clinical Bureaucracy KPI Dashboard")
    st.caption("Monitoraggio EHR: documentazione, review, ordini, inbox, after-hours e impatto note AI.")

    mode = st.sidebar.radio("Tipo dati", ["Sintetici (demo)", "Carica CSV", "Carica PDF"])

    df = None
    if mode == "Sintetici (demo)":
        n_visits = st.sidebar.slider("Numero visite", 50, 2000, 400, step=50)
        n_clin = st.sidebar.slider("Numero medici", 3, 40, 12, step=1)
        seed = st.sidebar.number_input("Seed", 0, 10_000, 42)
        df = create_synthetic_logs(n_visits=n_visits, n_clinicians=n_clin, seed=seed)

    elif mode == "Carica CSV":
        f = st.sidebar.file_uploader("Carica CSV", type=["csv"])
        if f is not None:
            df = load_csv(f)
        else:
            st.info("Carica un CSV con colonne: visit_id, clinician_id, department, activity, start_time, end_time, minutes, is_after_hours, is_ai_note, ai_edit_minutes")
            st.stop()
            
    elif mode == "Carica PDF":
        f = st.sidebar.file_uploader("Carica PDF", type=["pdf"])
        if f is not None:
            with st.spinner("Estrazione tabelle dal PDF in corso..."):
                df = load_pdf(f)
                if df.empty:
                    st.warning("Nessuna tabella trovata nel PDF o formato non supportato.")
                    st.stop()
                else:
                    st.success(f"Trovate e caricate {len(df)} righe dal PDF.")
        else:
            st.info("Carica un file PDF contenente tabelle con i dati clinici.")
            st.stop()

    if df is None:
        st.stop()

    # --- Filtri comuni ---
    st.sidebar.subheader("Filtri Reparto")
    aree_principali = ["Tutte le aree"] + list(REPARTI.keys())
    area_selezionata = st.sidebar.selectbox("Area Principale", aree_principali)

    if area_selezionata == "Tutte le aree":
        lista_reparti = ["Tutti i reparti"] + sorted([reparto for sublist in REPARTI.values() for reparto in sublist])
    else:
        lista_reparti = ["Tutti i reparti"] + sorted(REPARTI[area_selezionata])
    
    reparto_selezionato = st.sidebar.selectbox("Reparto Specifico", lista_reparti)

    if reparto_selezionato != "Tutti i reparti":
        df = df[df["department"] == reparto_selezionato]
    elif area_selezionata != "Tutte le aree":
        df = df[df["department"].isin(REPARTI[area_selezionata])]

    st.sidebar.subheader("Filtri Clinico")
    if 'clinician_id' in df.columns:
        clinicians = sorted(df["clinician_id"].unique())
        selected_clin = st.sidebar.multiselect("Filtra per clinico", options=clinicians, default=clinicians)
        df = df[df["clinician_id"].isin(selected_clin)]

    # --- KPI cards ---
    kpi = kpi_overview(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("⏱️ Min/visita (medio)", f"{kpi['avg_minutes_per_visit']:.2f}")
    c2.metric("🌙 After-hours min/visita", f"{kpi['avg_after_hours_minutes_per_visit']:.2f}")
    c3.metric("🤖 % visite con nota AI", f"{kpi['ai_note_share_percent']:.2f}%")
    c4.metric("✍️ Min correzione AI (medio)", f"{kpi['ai_correction_avg_minutes']:.2f}")

    st.divider()

    # --- Distribuzione per attività ---
    st.subheader("Distribuzione tempo per attività")
    act = share_time_by_activity(df)
    fig1, ax1 = plt.subplots()
    ax1.bar(act.index, act["minutes"])
    ax1.set_xlabel("Attività")
    ax1.set_ylabel("Minuti totali")
    ax1.set_title("Tempo totale per attività")
    st.pyplot(fig1, use_container_width=True)

    # --- Carico per clinico ---
    st.subheader("Carico per clinico (minuti totali)")
    cl = clinicians_workload(df)
    fig2, ax2 = plt.subplots()
    ax2.bar(cl["clinician_id"], cl["total_minutes"])
    ax2.set_xlabel("Clinico")
    ax2.set_ylabel("Minuti totali")
    ax2.set_title("Workload totale (ordinato)")
    plt.xticks(rotation=45)
    st.pyplot(fig2, use_container_width=True)

    # --- Outlier ---
    st.subheader("Visite outlier (durata totale elevata)")
    out = outlier_visits(df)
    st.dataframe(out)

    # --- Download ---
    st.download_button(
        "⬇️ Scarica dataset (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="clinical_logs.csv",
        mime="text/csv",
    )
    st.download_button(
        "⬇️ Scarica aggregati per attività (CSV)",
        data=act.to_csv().encode("utf-8"),
        file_name="activity_aggregates.csv",
        mime="text/csv",
    )
elif dataset_type == "Carica CSV Burocrazia":
    st.title("🩺 Clinical Bureaucracy KPI Dashboard (CSV Upload)")
    st.caption("Monitoraggio EHR: documentazione, review, ordini, inbox, after-hours e impatto note AI.")

    f = st.sidebar.file_uploader("Carica CSV", type=["csv"])
    if f is not None:
        df = load_csv(f)
    else:
        st.info("Carica un CSV con colonne: visit_id, clinician_id, department, activity, start_time, end_time, minutes, is_after_hours, is_ai_note, ai_edit_minutes")
        st.stop()

    # --- Filtri comuni ---
    st.sidebar.subheader("Filtri Reparto")
    aree_principali = ["Tutte le aree"] + list(REPARTI.keys())
    area_selezionata = st.sidebar.selectbox("Area Principale", aree_principali)

    if area_selezionata == "Tutte le aree":
        lista_reparti = ["Tutti i reparti"] + sorted([reparto for sublist in REPARTI.values() for reparto in sublist])
    else:
        lista_reparti = ["Tutti i reparti"] + sorted(REPARTI[area_selezionata])
    
    reparto_selezionato = st.sidebar.selectbox("Reparto Specifico", lista_reparti)

    if reparto_selezionato != "Tutti i reparti":
        df = df[df["department"] == reparto_selezionato]
    elif area_selezionata != "Tutte le aree":
        df = df[df["department"].isin(REPARTI[area_selezionata])]

    st.sidebar.subheader("Filtri Clinico")
    if 'clinician_id' in df.columns:
        clinicians = sorted(df["clinician_id"].unique())
        selected_clin = st.sidebar.multiselect("Filtra per clinico", options=clinicians, default=clinicians)
        df = df[df["clinician_id"].isin(selected_clin)]

    # --- KPI cards ---
    kpi = kpi_overview(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("⏱️ Min/visita (medio)", f"{kpi['avg_minutes_per_visit']:.2f}")
    c2.metric("🌙 After-hours min/visita", f"{kpi['avg_after_hours_minutes_per_visit']:.2f}")
    c3.metric("🤖 % visite con nota AI", f"{kpi['ai_note_share_percent']:.2f}%")
    c4.metric("✍️ Min correzione AI (medio)", f"{kpi['ai_correction_avg_minutes']:.2f}")

    st.divider()

    # --- Distribuzione per attività ---
    st.subheader("Distribuzione tempo per attività")
    act = share_time_by_activity(df)
    fig1, ax1 = plt.subplots()
    ax1.bar(act.index, act["minutes"])
    ax1.set_xlabel("Attività")
    ax1.set_ylabel("Minuti totali")
    ax1.set_title("Tempo totale per attività")
    st.pyplot(fig1, use_container_width=True)

    # --- Carico per clinico ---
    st.subheader("Carico per clinico (minuti totali)")
    cl = clinicians_workload(df)
    fig2, ax2 = plt.subplots()
    ax2.bar(cl["clinician_id"], cl["total_minutes"])
    ax2.set_xlabel("Clinico")
    ax2.set_ylabel("Minuti totali")
    ax2.set_title("Workload totale (ordinato)")
    plt.xticks(rotation=45)
    st.pyplot(fig2, use_container_width=True)

    # --- Outlier ---
    st.subheader("Visite outlier (durata totale elevata)")
    out = outlier_visits(df)
    st.dataframe(out)

    # --- Download ---
    st.download_button(
        "⬇️ Scarica dataset (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="clinical_logs.csv",
        mime="text/csv",
    )
    st.download_button(
        "⬇️ Scarica aggregati per attività (CSV)",
        data=act.to_csv().encode("utf-8"),
        file_name="activity_aggregates.csv",
        mime="text/csv",
    )
else:
    st.error("Selezione non valida.")