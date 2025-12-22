import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os
import shutil

# Konfigurasi Path
DATA_PATH = "dataset.csv"
MODEL_PATH = "model_output"

def main():
    print("--- Memulai Training CI/CD ---")
    
    # Load Data
    if not os.path.exists(DATA_PATH):
        print("Dataset tidak ditemukan!")
        return

    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=['passed_exam']) # Sesuaikan target
    y = df['passed_exam']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Hapus folder model lama jika ada (biar bersih)
    if os.path.exists(MODEL_PATH):
        shutil.rmtree(MODEL_PATH)

    # Start Run
    mlflow.set_experiment("CI_Experiment")
    
    with mlflow.start_run() as run:
        # Autologging
        mlflow.sklearn.autolog()

        # Train
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)

        # Evaluasi
        acc = accuracy_score(y_test, clf.predict(X_test))
        print(f"Accuracy: {acc}")

        # --- PENTING UNTUK DOCKER ---
        # Save model ke folder lokal agar bisa dibuild jadi image
        mlflow.sklearn.save_model(clf, MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()