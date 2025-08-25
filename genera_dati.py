import json
import random
from datetime import datetime, timedelta

# --- CONFIGURAZIONE ---
NUM_PAZIENTI = 400
NUM_ADMISSIONS = 1200
OUTPUT_FILENAME = "simulated_ricoveri_arricchito.json"
# --------------------

def generate_patient_pool(num_patients):
    """Genera un pool di pazienti unici."""
    patients = []
    first_names_m = ["Marco", "Paolo", "Andrea", "Giovanni", "Giuseppe", "Raffaele", "Luca", "Alessandro", "Davide"]
    first_names_f = ["Giulia", "Chiara", "Maria", "Laura", "Sara", "Francesca", "Elena"]
    last_names = ["Romano", "Russo", "Marino", "Rossi", "Ferrari", "Bianchi", "Gallo", "Giordano", "Ricci", "Verdi", "Esposito"]
    
    for i in range(1, num_patients + 1):
        sex = random.choice(["M", "F"])
        if sex == "M":
            name = f"{random.choice(first_names_m)} {random.choice(last_names)}"
        else:
            name = f"{random.choice(first_names_f)} {random.choice(last_names)}"
        
        patients.append({
            "patient_id": f"P{i:04d}",
            "patient_name": name,
            "age": random.randint(18, 95),
            "sex": sex
        })
    print(f"✅ Creato un pool di {len(patients)} pazienti.")
    return patients

def generate_admission_data(admission_id, patient_data):
    """
    Genera i dati di un singolo ricovero con logica clinica correlata.
    """
    groups = ["Medicina", "Chirurgia", "Riabilitazione", "Maternita_Pediatria", "Neurologia", "Sensoriali_Dermato"]
    departments = {
        "Medicina": ["Cardiologia", "Gastroenterologia", "Endocrinologia", "Nefrologia", "Geriatria", "Medicina Interna", "Medicina d'Urgenza", "Pneumologia"],
        "Chirurgia": ["Chirurgia Generale", "Chirurgia Toracica", "Chirurgia Vascolare", "Chirurgia Plastica", "Ortopedia e Traumatologia", "Neurochirurgia"],
        "Riabilitazione": ["Fisioterapia", "Logopedia", "Riabilitazione Generale"],
        "Maternita_Pediatria": ["Sala Parto", "Ostetricia e Ginecologia", "Pediatria"],
        "Neurologia": ["Neurologia", "Neuropsichiatria Infantile"],
        "Sensoriali_Dermato": ["Dermatologia", "Otorinolaringoiatria (ORL)"]
    }
    diagnoses = {
        "Polmonite": {"group": "Medicina", "base_los": 8},
        "Insufficienza Renale": {"group": "Medicina", "base_los": 7},
        "Frattura": {"group": "Chirurgia", "base_los": 5},
        "Diabete": {"group": "Medicina", "base_los": 4},
        "Ipertensione": {"group": "Medicina", "base_los": 3},
        "Neoplasia": {"group": "Chirurgia", "base_los": 12},
        "Ictus": {"group": "Neurologia", "base_los": 10},
        "Asma": {"group": "Medicina", "base_los": 3},
        "Parto": {"group": "Maternita_Pediatria", "base_los": 3},
        "Riabilitazione post-op": {"group": "Riabilitazione", "base_los": 15}
    }
    comorbidities_options = ["Diabete", "Ipertensione", "Insufficienza Renale", "BPCO", "Fibrillazione Atriale", "Obesità"]
    
    # --- Selezione coerente di diagnosi e reparto ---
    diagnosis = random.choice(list(diagnoses.keys()))
    group = diagnoses[diagnosis]["group"]
    
    # Logica per Maternità/Pediatria
    if group == "Maternita_Pediatria":
        if patient_data["sex"] == "M" or patient_data["age"] > 45:
            diagnosis = "Frattura" # Diagnosi più generica
            group = "Chirurgia"
    department = random.choice(departments[group])

    # --- Generazione Comorbidità Robusta ---
    num_comorbidities = random.choices([0, 1, 2, 3], weights=[0.3, 0.4, 0.2, 0.1], k=1)[0]
    if num_comorbidities == 0:
        comorbidities = "None"
    else:
        comorbidities = ";".join(random.sample(comorbidities_options, num_comorbidities))

    # --- Generazione Durata Ricovero (length_days) CORRELATA ---
    los = diagnoses[diagnosis]["base_los"]
    severity = random.choices(["low", "moderate", "high"], weights=[0.5, 0.35, 0.15], k=1)[0]

    if severity == 'moderate': los += random.randint(2, 6)
    if severity == 'high': los += random.randint(5, 15)
    if patient_data['age'] > 75: los += random.randint(1, 5)
    los += num_comorbidities * random.randint(1, 3)
    
    intervento_chirurgico = (group == "Chirurgia" and random.random() > 0.2)
    if intervento_chirurgico: los += random.randint(3, 7)
    
    los += random.randint(-2, 2) # Aggiunge un po' di rumore
    length_days = max(1, int(los)) # Assicura che sia almeno 1 giorno

    admission_date = datetime.now() - timedelta(days=random.randint(1, 1095))
    discharge_date = admission_date + timedelta(days=length_days)

    # --- Generazione Valori Clinici CORRELATI ---
    # Valori di base per un paziente sano
    pressione_sistolica = random.randint(115, 130)
    pressione_diastolica = random.randint(75, 85)
    frequenza_cardiaca = random.randint(65, 85)
    saturazione_ossigeno = random.randint(96, 99)
    livello_creatinina = round(random.uniform(0.7, 1.1), 1)
    globuli_bianchi = random.randint(5000, 9000)
    indice_pcr = round(random.uniform(1.0, 8.0), 1)

    # Alterazioni basate sulla diagnosi
    if diagnosis in ["Polmonite", "Infezione"]:
        saturazione_ossigeno = random.randint(90, 95)
        globuli_bianchi = random.randint(11000, 18000)
        indice_pcr = random.uniform(50, 150)
    elif diagnosis == "Insufficienza Renale" or "Insufficienza Renale" in comorbidities:
        livello_creatinina = random.uniform(1.5, 3.5)
        pressione_sistolica += random.randint(10, 20)
    elif diagnosis in ["Ictus", "Infarto"]:
        pressione_sistolica = random.randint(150, 190)
        frequenza_cardiaca = random.randint(90, 115)
        indice_pcr = random.uniform(20, 60)
    elif diagnosis == "Ipertensione" or "Ipertensione" in comorbidities:
        pressione_sistolica += random.randint(15, 30)
        pressione_diastolica += random.randint(5, 15)
    
    if intervento_chirurgico:
        indice_pcr += random.uniform(20, 50)
        globuli_bianchi += random.randint(1000, 4000)

    # Alterazioni basate sulla gravità
    if severity == 'moderate':
        pressione_sistolica += random.randint(5, 10)
        frequenza_cardiaca += random.randint(5, 10)
        indice_pcr *= 1.2
    elif severity == 'high':
        pressione_sistolica += random.randint(10, 25)
        frequenza_cardiaca += random.randint(10, 20)
        saturazione_ossigeno = max(88, saturazione_ossigeno - 5)
        indice_pcr *= 1.8
        globuli_bianchi += random.randint(2000, 6000)
        livello_creatinina *= 1.2

    return {
        "admission_id": admission_id,
        "patient_id": patient_data["patient_id"],
        "patient_name": patient_data["patient_name"],
        "age": patient_data["age"],
        "sex": patient_data["sex"],
        "group": group,
        "department": department,
        "admission_date": admission_date.strftime("%Y-%m-%d"),
        "discharge_date": discharge_date.strftime("%Y-%m-%d"),
        "length_days": length_days,
        "diagnosis": diagnosis,
        "comorbidities": comorbidities,
        "severity": severity,
        "prior_admissions": random.randint(0, 5),
        "from_emergency": random.random() < 0.4, # 40% di probabilità
        "ai_note": random.random() < 0.3, # 30% di probabilità
        "pressione_sistolica": int(pressione_sistolica),
        "pressione_diastolica": int(pressione_diastolica),
        "frequenza_cardiaca": int(frequenza_cardiaca),
        "saturazione_ossigeno": int(saturazione_ossigeno),
        "livello_creatinina": round(livello_creatinina, 1),
        "globuli_bianchi": int(globuli_bianchi),
        "indice_pcr": round(indice_pcr, 1),
        "intervento_chirurgico": intervento_chirurgico
    }

def main():
    """Funzione principale per generare il file JSON completo."""
    patient_pool = generate_patient_pool(NUM_PAZIENTI)
    all_admissions = []
    
    for i in range(1, NUM_ADMISSIONS + 1):
        admission_id = f"A{i:05d}"
        random_patient = random.choice(patient_pool)
        admission_data = generate_admission_data(admission_id, random_patient)
        all_admissions.append(admission_data)
        
    with open(OUTPUT_FILENAME, "w", encoding="utf-8") as f:
        json.dump(all_admissions, f, indent=4, ensure_ascii=False)
    
    print(f"✅ File '{OUTPUT_FILENAME}' con {len(all_admissions)} ricoveri generato con successo!")

if __name__ == "__main__":
    main()