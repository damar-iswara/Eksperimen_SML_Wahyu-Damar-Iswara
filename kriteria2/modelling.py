import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# --- KONFIGURASI ---
DATA_PATH = "kriteria1\preprocessing\python_learning_exam_performance_preprocessing.csv"
TARGET_COL = "passed_exam"

def main():
    print("Membaca data...")
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"Error: File {DATA_PATH} tidak ditemukan di folder ini.")
        return

    # Pisahkan Fitur (X) dan Target (y)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # Split Data (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Aktifkan Autolog
    mlflow.sklearn.autolog()

    # Set Eksperimen
    mlflow.set_experiment("Modelling_Basic")

    # Mulai Training
    print("Memulai training model...")
    with mlflow.start_run(run_name="Basic_RandomForest_NoTuning"):
        model = RandomForestClassifier(random_state=42)
        
        model.fit(X_train, y_train)
        
        # Evaluasi (Opsional print di terminal, tapi sudah tercatat di MLflow)
        acc = accuracy_score(y_test, model.predict(X_test))
        print(f"Training Selesai. Akurasi: {acc}")
        print("Cek hasil lengkap di MLflow UI.")

if __name__ == "__main__":
    main()