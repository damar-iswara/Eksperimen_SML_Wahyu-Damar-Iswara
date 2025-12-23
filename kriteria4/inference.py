import requests
import time
import pandas as pd
import json

# URL Endpoint Flask
URL = "http://localhost:5000/predict"

# Path ke file CSV hasil preprocessing (yg dipakai training)
# Sesuaikan path ini kalau filenya ada di folder lain
DATA_PATH = "../kriteria3/python_learning_exam_performance_preprocessing.csv"

def main():
    print(f"Loading data template from: {DATA_PATH}")
    try:
        # Load dataset asli
        df_source = pd.read_csv(DATA_PATH)
        
        # Buang kolom target (passed_exam) karena kita mau prediksi
        if 'passed_exam' in df_source.columns:
            df_source = df_source.drop(columns=['passed_exam'])
            
        print(f"Data loaded. Columns ({len(df_source.columns)}): {list(df_source.columns)}")
        print("--- Mulai mengirim traffic data dummy (Sampling dari CSV asli) ---")
        
        while True:
            # Ambil 1 baris acak dari dataset
            random_row = df_source.sample(1)
            
            # Ubah jadi format JSON/Dictionary
            payload = random_row.to_dict(orient='records')
            
            # Kirim request
            try:
                response = requests.post(URL, json=payload)
                if response.status_code == 200:
                    print(f"Success [200]: Prediksi {response.json()['prediction']}")
                else:
                    print(f"Failed [{response.status_code}]: {response.text}")
            except requests.exceptions.ConnectionError:
                print("Error: Koneksi ke Server Flask terputus (Cek terminal prometheus_exporter.py)")

            # Tidur sebentar (0.5 - 2 detik)
            time.sleep(1)

    except FileNotFoundError:
        print(f"ERROR: File {DATA_PATH} tidak ditemukan!")
        print("Pastikan file csv hasil preprocessing ada di lokasi yang benar.")

if __name__ == "__main__":
    main()