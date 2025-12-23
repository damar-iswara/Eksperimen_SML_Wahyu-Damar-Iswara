import time
import psutil
import mlflow.sklearn
import pandas as pd
from flask import Flask, request, jsonify
from prometheus_client import start_http_server, Counter, Gauge, Histogram, Summary

# --- KONFIGURASI ---
MODEL_PATH = "kriteria3\model_output"

app = Flask(__name__)

# --- LOAD MODEL ---
print("Loading model...")
model = mlflow.sklearn.load_model(MODEL_PATH)
print("Model loaded!")

# --- 10 METRIKS (SYARAT ADVANCE) ---

# 1. Counter: Total Request masuk
REQUEST_COUNT = Counter('app_request_count', 'Total request received')
# 2. Counter: Request sukses (HTTP 200)
SUCCESS_COUNT = Counter('app_success_count', 'Total successful predictions')
# 3. Counter: Request gagal (HTTP 500/400)
FAILURE_COUNT = Counter('app_failure_count', 'Total failed predictions')

# 4. Histogram: Latency / Waktu proses (detik)
LATENCY = Histogram('app_latency_seconds', 'Time query processing')

# 5. Gauge: Prediksi 'Lulus' (Class 1)
PRED_CLASS_1 = Gauge('app_prediction_class_1', 'Count of Class 1 predictions')
# 6. Gauge: Prediksi 'Gagal' (Class 0)
PRED_CLASS_0 = Gauge('app_prediction_class_0', 'Count of Class 0 predictions')

# 7. Gauge: CPU Usage Sistem
SYSTEM_CPU = Gauge('system_cpu_usage', 'Current CPU usage percent')
# 8. Gauge: Memory Usage Sistem
SYSTEM_MEMORY = Gauge('system_memory_usage', 'Current RAM usage percent')

# 9. Summary: Ukuran Data Input (bytes)
INPUT_SIZE = Summary('app_input_size_bytes', 'Size of input payload')

# 10. Gauge: Jumlah Baris Data per Request
ROW_COUNT = Gauge('app_row_count', 'Number of rows in one request')


def update_system_metrics():
    """Update metriks CPU & RAM"""
    SYSTEM_CPU.set(psutil.cpu_percent())
    SYSTEM_MEMORY.set(psutil.virtual_memory().percent)

@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    REQUEST_COUNT.inc()
    
    try:
        # Update system metrics setiap request
        update_system_metrics()
        
        # Terima Data
        data = request.json
        df = pd.DataFrame(data)
        
        # Catat ukuran data & jumlah baris
        INPUT_SIZE.observe(len(str(data)))
        ROW_COUNT.set(len(df))

        # Prediksi
        predictions = model.predict(df)
        
        # Hitung Kelas Prediksi
        class_1 = sum(predictions)
        class_0 = len(predictions) - class_1
        
        PRED_CLASS_1.set(class_1)
        PRED_CLASS_0.set(class_0)
        
        SUCCESS_COUNT.inc()
        
        # Catat Latency
        process_time = time.time() - start_time
        LATENCY.observe(process_time)

        return jsonify({'prediction': predictions.tolist()})

    except Exception as e:
        FAILURE_COUNT.inc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Server metrics Prometheus di port 8000
    start_http_server(8000)

    # Server Model Flask di port 5000
    app.run(host='0.0.0.0', port=5000)