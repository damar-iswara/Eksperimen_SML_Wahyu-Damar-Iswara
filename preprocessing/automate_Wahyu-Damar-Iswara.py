import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

def perform_preprocessing(input_path, output_path):
    """
    Fungsi ini melakukan cleaning, transformasi, dan feature engineering
    """
    print("--- Memulai Proses Otomatisasi ---")

    # 1. Load Data
    if not os.path.exists(input_path):
        print(f"Error: File input tidak ditemukan di {input_path}")
        return

    df = pd.read_csv(input_path)
    print(f"Data dimuat. Shape awal: {df.shape}")

    # 2. Drop ID Column
    if 'student_id' in df.columns:
        df = df.drop(columns=['student_id'])

    # 3. Missing Values Handling
    # Spesifik untuk 'prior_programming_experience'
    if 'prior_programming_experience' in df.columns:
        df['prior_programming_experience'] = df['prior_programming_experience'].fillna('No Experience')

    # Identifikasi kolom numerik dan kategorikal awal
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns

    # Isi NaN numerik dengan Mean
    for col in num_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mean())

    # Isi NaN kategorikal dengan Modus
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            # Menggunakan mode()[0] untuk mengambil modus pertama
            df[col] = df[col].fillna(df[col].mode()[0])

    # 4. Duplicate Handling
    initial_rows = len(df)
    df.drop_duplicates(inplace=True)
    print(f"Baris duplikat dihapus: {initial_rows - len(df)}")

    # 5. Binning (Age -> Age Group)
    if 'age' in df.columns:
        bins = [15, 18, 24, 34, 44, 55]
        labels = ['Teen', 'Young Adult', 'Adult', 'Mid Adult', 'Senior Adult']

        df['age_group'] = pd.cut(
            df['age'],
            bins=bins,
            labels=labels
        )
        # Drop kolom asli 'age' setelah binning sesuai eksperimen
        df = df.drop(columns=['age'])

    # 6. Scaling (MinMaxScaler)
    # Re-identifikasi numerik karena 'age' sudah hilang
    num_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    if num_columns:
        scaler = MinMaxScaler()
        df[num_columns] = scaler.fit_transform(df[num_columns])

    # 7. Encoding (LabelEncoder)
    # Re-identifikasi kategorikal karena ada 'age_group' baru
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    for col in cat_cols:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col])

    # 8. Outlier Handling (IQR Clipping)
    # Re-identifikasi numerik untuk proses outlier
    num_cols_final = df.select_dtypes(include=['int64', 'float64']).columns

    Q1 = df[num_cols_final].quantile(0.25)
    Q3 = df[num_cols_final].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    for feature in num_cols_final:
        if feature == 'passed_exam':
            continue # Skip target variable

        df[feature] = df[feature].clip(
            lower=lower_bound[feature],
            upper=upper_bound[feature]
        )

    # 9. Save Data
    folder_path = os.path.dirname(output_path)
    
    # Hanya buat folder jika folder_path tidak kosong
    if folder_path:
        os.makedirs(folder_path, exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"Proses selesai! Data disimpan di: {output_path}")
    print(f"Shape akhir: {df.shape}")

if __name__ == "__main__":
    # Path relative ke file raw di folder sibling
    INPUT_FILE = "../dataset/python_learning_exam_performance.csv"
    
    # Path output
    OUTPUT_FILE = "python_learning_exam_performance_preprocessing.csv"
    
    perform_preprocessing(INPUT_FILE, OUTPUT_FILE)