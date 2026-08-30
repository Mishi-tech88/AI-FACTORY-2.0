# src/data_generation.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os
import cv2
from PIL import Image
from fpdf import FPDF

# ------------------ CREATE DIRECTORIES ------------------
os.makedirs('data/raw', exist_ok=True)
os.makedirs('data/images', exist_ok=True)

def generate_all_data():
    # ========== 1. TABULAR PRODUCTION RECORDS ==========
    num_days = 90
    machines = ['M1','M2','M3','M4']
    start_date = datetime(2025,1,1)
    hours_per_day = 24

    timestamps = [start_date + timedelta(hours=i) for i in range(num_days * hours_per_day)]
    records = []

    for ts in timestamps:
        for machine in machines:
            prod_count = np.random.poisson(lam=50)
            quality = np.random.beta(2,5)
            temp = 70 + 10*np.sin(2*np.pi*(ts.hour)/24) + np.random.normal(0,2)
            operator = random.choice(['OpA','OpB','OpC'])
            records.append([ts, machine, prod_count, quality, temp, operator])

    df_prod = pd.DataFrame(records, columns=['timestamp','machine_id','production_count','quality_score','temperature','operator'])
    # Introduce missing values (5% random)
    for col in ['production_count','quality_score','temperature']:
        df_prod.loc[df_prod.sample(frac=0.05).index, col] = np.nan
    # Duplicates
    df_prod = pd.concat([df_prod, df_prod.sample(frac=0.02)], ignore_index=True)
    df_prod.to_csv('data/raw/production.csv', index=False)

    # ========== 2. TIME-SERIES SENSOR READINGS ==========
    sensor_timestamps = []
    sensor_data = []
    for ts in timestamps:
        for minute in [0,10,20,30,40,50]:
            t = ts + timedelta(minutes=minute)
            for machine in machines:
                vibration = 0.5 + 0.3*np.sin(2*np.pi*t.hour/24) + np.random.normal(0,0.1)
                pressure = 100 + 5*np.sin(2*np.pi*t.hour/24) + np.random.normal(0,1)
                rpm = 3000 + 200*np.sin(2*np.pi*t.hour/24) + np.random.normal(0,50)
                if random.random() < 0.01:
                    vibration += 2.0
                sensor_timestamps.append(t)
                sensor_data.append([machine, vibration, pressure, rpm])

    df_sensor = pd.DataFrame(sensor_data, columns=['machine_id','vibration','pressure','rpm'])
    df_sensor['timestamp'] = sensor_timestamps
    for col in ['vibration','pressure','rpm']:
        df_sensor.loc[df_sensor.sample(frac=0.03).index, col] = np.nan
    df_sensor.to_csv('data/raw/sensors.csv', index=False)

    # ========== 3. IMAGE DATA (defect detection) ==========
    image_records = []
    for i in range(1000):
        img = np.random.normal(100, 30, (64,64)).astype(np.uint8)
        is_defect = random.random() < 0.1
        if is_defect:
            cv2.circle(img, (random.randint(10,54), random.randint(10,54)), 5, 255, -1)
        img_path = f'data/images/prod_{i}.png'
        Image.fromarray(img).save(img_path)
        image_records.append([f'prod_{i}.png', is_defect, random.choice(machines)])

    df_images = pd.DataFrame(image_records, columns=['image_path','defect_flag','machine_id'])
    df_images.to_csv('data/raw/image_metadata.csv', index=False)

    # ========== 4. TEXT MAINTENANCE NOTES ==========
    text_records = []
    for _ in range(200):
        machine = random.choice(machines)
        date = start_date + timedelta(days=random.randint(0,num_days-1))
        issue = random.choice(['oil leak','overheating','unusual vibration','noise','error code 42'])
        action = random.choice(['replaced filter','cleaned sensor','tightened bolts','reset controller'])
        note = f"Machine {machine}: {issue} detected. {action}. Operator: {random.choice(['John','Mary','Tom'])}."
        text_records.append([machine, date, note])

    df_text = pd.DataFrame(text_records, columns=['machine_id','date','maintenance_note'])
    df_text.to_csv('data/raw/maintenance_notes.csv', index=False)

    # ========== 5. PDF MANUALS / SOPs ==========
    pdf_texts = {
        'data/raw/M1_manual.pdf': "M1 Operating Manual: Keep temperature below 80°C. Regular lubrication every 500 hours.",
        'data/raw/SOP_general.pdf': "Standard Operating Procedure: Always wear safety gear. In case of error, reboot controller."
    }
    for fname, content in pdf_texts.items():
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, content)
        pdf.output(fname)

    print("All data generated successfully! Check 'data/raw/' and 'data/images/'.")

if __name__ == "__main__":
    generate_all_data()