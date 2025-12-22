import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import dagshub
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# --- KONFIGURASI ---
DATA_PATH = "kriteria2\python_learning_exam_performance_preprocessing.csv"
TARGET_COL = "passed_exam"

DAGSHUB_USERNAME = "damar-iswara"
DAGSHUB_REPO_NAME = "Eksperimen_SML_Wahyu-Damar-Iswara"

def main():
    # Setup DagsHub
    dagshub.init(repo_owner=DAGSHUB_USERNAME, repo_name=DAGSHUB_REPO_NAME, mlflow=True)
    
    # Set Eksperimen
    mlflow.set_experiment("Eksperimen_Advance_Tuning")

    print("--- Memuat Data ---")
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Hyperparameter Tuning
    print("--- Memulai Tuning (GridSearch) ---")
    rf = RandomForestClassifier(random_state=42)
    
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print(f"Parameter Terbaik: {best_params}")

    # Evaluasi Model
    y_pred = best_model.predict(X_test)
    
    # Hitung Metriks secara Manual
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print("--- Mengirim Log ke DagsHub ---")
    with mlflow.start_run(run_name="Tuned_RandomForest_Advance"):
        
        # MANUAL LOGGING PARAMETER
        mlflow.log_params(best_params)
        
        # MANUAL LOGGING METRIKS
        metrics = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1
        }
        mlflow.log_metrics(metrics)
        
        # C. LOG MODEL
        mlflow.sklearn.log_model(best_model, "best_model")

        # ARTEFAK TAMBAHAN
        # Confusion Matrix Image
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig("confusion_matrix.png")
        plt.close()
        
        mlflow.log_artifact("confusion_matrix.png")
        print("Artefak 1 (Confusion Matrix) terupload.")

        # Feature Importance Image
        if hasattr(best_model, 'feature_importances_'):
            plt.figure(figsize=(10, 6))
            importances = best_model.feature_importances_
            indices = np.argsort(importances)[::-1]
            features = X.columns
            
            sns.barplot(x=importances[indices], y=features[indices])
            plt.title('Feature Importances')
            plt.tight_layout()
            plt.savefig("feature_importance.png")
            plt.close()
            
            mlflow.log_artifact("feature_importance.png") # Upload ke DagsHub
            print("Artefak 2 (Feature Importance) terupload.")

    print(f"Selesai! Cek hasil di: https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO_NAME}")

if __name__ == "__main__":
    main()