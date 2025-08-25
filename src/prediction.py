import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import joblib
import os


# Importazione dei nuovi modelli da testare
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
import xgboost as xgb
import lightgbm as lgb

def load_and_preprocess_data(json_path: str):
    """
    Carica i dati, esegue una pulizia robusta e crea nuove features 
    per migliorare l'accuratezza del modello.
    """
    if not os.path.exists(json_path):
        print(f"Errore: Il file '{json_path}' non è stato trovato.")
        return pd.DataFrame()
        
    df = pd.read_json(json_path)

    # --- INGEGNERIA DELLE CARATTERISTICHE (PULIZIA MIGLIORATA) ---

    # 1. Funzione robusta per contare le comorbidità reali
    def conta_comorbidita_reali(comorbidities_str):
        if pd.isna(comorbidities_str):
            return 0
        
        # Sostituisce virgole con punto e virgola per standardizzare il separatore
        items_str = str(comorbidities_str).replace(',', ';')
        items = [item.strip().lower() for item in items_str.split(';')]
        
        # Filtra via le parole che non indicano una malattia
        items_validi = [item for item in items if item not in ['none', 'nessuna', '']]
        
        return len(set(items_validi)) # Usa set per contare solo comorbidità uniche

    df['numero_comorbidita'] = df['comorbidities'].apply(conta_comorbidita_reali)

    # 2. Severità (Ordinal Encoding)
    severity_map = {'low': 0, 'moderate': 1, 'high': 2}
    df['severita_numerica'] = df['severity'].map(severity_map)

    # 3. Ingresso da PS (Boolean to Integer)
    df['da_ps'] = df['from_emergency'].astype(int)

    # 4. Rinomina le colonne per chiarezza
    df.rename(columns={
        "age": "età", "sex": "sesso", "diagnosis": "diagnosi_principale",
        "department": "reparto", "length_days": "giorni_ricovero",
        "prior_admissions": "ricoveri_precedenti"
    }, inplace=True)

    # 5. Gestione delle date e pulizia finale
    df['data_ammissione'] = pd.to_datetime(df['admission_date'], errors='coerce')
    df['data_dimissione'] = pd.to_datetime(df['discharge_date'], errors='coerce')
    
    # Rimuove righe dove le informazioni essenziali sono mancanti
    df.dropna(subset=['severita_numerica', 'giorni_ricovero', 'età'], inplace=True)
    df['severita_numerica'] = df['severita_numerica'].astype(int)
    
    print("✅ Dati caricati e pre-elaborati con successo.")
    return df

def train_evaluate_and_save_best_model(df, model_path="modello_dimissione.joblib"):
    """
    Addestra più modelli, li valuta, stampa un confronto e salva il migliore.
    """
    # 1. Seleziona le caratteristiche (X) e la variabile target (y)
    # Escludiamo le colonne non necessarie o che causerebbero data leak
    features_to_drop = [
        "admission_id", "patient_id", "patient_name", "group", "admission_date", 
        "discharge_date", "comorbidities", "severity", "from_emergency", 
        "ai_note", "giorni_ricovero", "data_ammissione", "data_dimissione"
    ]
    X = df.drop(columns=features_to_drop, errors='ignore')
    y = df["giorni_ricovero"]

    # 2. Suddivide i dati in set di addestramento e di test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Definisce il preprocessore
    # Identifica automaticamente le colonne numeriche e categoriche
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ],
        remainder='drop'
    )

    # 4. Definisce i modelli da testare
    models = {
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
        "XGBoost": xgb.XGBRegressor(n_estimators=100, random_state=42, eval_metric='rmse'),
        "LightGBM": lgb.LGBMRegressor(n_estimators=100, random_state=42),
        "Ridge": Ridge(alpha=1.0, random_state=42)
    }

    results = []
    best_model = None
    best_r2 = -np.inf

    # 5. Ciclo di addestramento e valutazione
    for name, regressor in models.items():
        # Crea la pipeline completa
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("regressor", regressor)
        ])
        
        # Addestra il modello
        pipeline.fit(X_train, y_train)
        
        # Valuta il modello
        y_pred = pipeline.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        results.append({"Modello": name, "R²": r2, "MAE": mae, "RMSE": rmse})
        
        # Controlla se è il modello migliore finora
        if r2 > best_r2:
            best_r2 = r2
            best_model = pipeline

    # 6. Stampa i risultati in una tabella di confronto
    results_df = pd.DataFrame(results).sort_values(by="R²", ascending=False)
    print("\n--- Confronto Performance Modelli ---")
    print(results_df.to_string(index=False))
    print("-------------------------------------\n")

    # 7. Salva il modello migliore
    if best_model:
        joblib.dump(best_model, model_path)
        best_model_name = results_df.iloc[0]['Modello']
        print(f"🏆 Modello migliore ('{best_model_name}') salvato con successo in '{model_path}'")
    else:
        print("⚠️ Nessun modello è stato addestrato con successo.")

def load_model_and_predict(input_data: pd.DataFrame, model_path="modello_dimissione.joblib"):
    """
    Carica il modello addestrato e restituisce la previsione.
    """
    if not os.path.exists(model_path):
        print(f"Errore: Modello non trovato in '{model_path}'")
        return None
    try:
        model = joblib.load(model_path)
        prediction = model.predict(input_data)
        return prediction[0]
    except Exception as e:
        print(f"Errore durante la previsione: {e}")
        return None


# --- BLOCCO DI ESECUZIONE DIRETTA ---
if __name__ == "__main__":
    # Assicurati di usare il file arricchito!
    file_json = "simulated_ricoveri.json"

    # 1. Carica e pre-elabora i dati
    df = load_and_preprocess_data(file_json)
    
    if not df.empty:
        # 2. Addestra, valuta e salva il modello migliore
        train_evaluate_and_save_best_model(df, model_path="modello_dimissione.joblib")
        
        # 3. Crea un paziente di esempio COMPLETO per testare la previsione
        #    con tutte le feature che il modello si aspetta.
        esempio_input = pd.DataFrame([{
            "età": 72,
            "sesso": "F",
            "diagnosi_principale": "Polmonite",
            "reparto": "Fisioterapia",
            "ricoveri_precedenti": 4,
            "da_ps": 1,
            "severita_numerica": 0, # low
            "numero_comorbidita": 2,
            "pressione_sistolica": 138,
            "pressione_diastolica": 85,
            "frequenza_cardiaca": 92,
            "saturazione_ossigeno": 94,
            "livello_creatinina": 1.5,
            "globuli_bianchi": 12800,
            "indice_pcr": 65.0,
            "intervento_chirurgico": False
        }])
        
        # 4. Carica il modello salvato (il migliore) e fai una previsione
        print("\n--- Test di Previsione su un Esempio ---")
        previsione = load_model_and_predict(esempio_input, model_path="modello_dimissione.joblib")
        
        if previsione is not None:
            print(f"Paziente di esempio: {esempio_input.iloc[0].to_dict()}")
            print(f"➡️ Previsione Durata Ricovero: {previsione:.2f} giorni")
        else:
            print("Impossibile effettuare la previsione di esempio.")
    else:
        print("Processo interrotto a causa di dati non disponibili.")