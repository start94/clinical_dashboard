import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_simulated_clinical_data(num_visits=100, num_clinicians=10):
    """
    Generates a simulated dataset for the Clinical Bureaucracy Dashboard.
    """
    activities = ["Documentation", "Review", "Orders", "Inbox"]
    departments = ["Cardiology", "Oncology", "Pediatrics", "Neurology", "General Medicine"]
    
    data = []
    
    clinicians = [f"Clinician_{i+1}" for i in range(num_clinicians)]
    
    for visit_id in range(num_visits):
        visit_clinician = np.random.choice(clinicians)
        visit_department = np.random.choice(departments)
        
        # Simulate a series of activities for each visit
        for activity in activities:
            start_time = datetime.now() - timedelta(minutes=np.random.randint(60, 180), seconds=np.random.randint(0, 59))
            minutes = np.random.randint(2, 25)
            end_time = start_time + timedelta(minutes=minutes)
            
            is_after_hours = np.random.choice([True, False], p=[0.15, 0.85])
            
            is_ai_note = False
            ai_edit_minutes = 0
            if activity == "Documentation" and np.random.random() > 0.6:  # 40% chance of AI note
                is_ai_note = True
                ai_edit_minutes = np.random.randint(1, 5)

            data.append([
                visit_id,
                visit_clinician,
                visit_department,
                activity,
                start_time,
                end_time,
                minutes,
                is_after_hours,
                is_ai_note,
                ai_edit_minutes
            ])

    df = pd.DataFrame(data, columns=[
        "visit_id",
        "clinician_id",
        "department",
        "activity",
        "start_time",
        "end_time",
        "minutes",
        "is_after_hours",
        "is_ai_note",
        "ai_edit_minutes"
    ])
    
    return df

# Create the dataframe and save to CSV
df = create_simulated_clinical_data()
df.to_csv("simulated_clinical_logs.csv", index=False)
print("Simulated dataset created and saved to 'simulated_clinical_logs.csv'")