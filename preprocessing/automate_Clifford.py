import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Ubah ke path lokal agar sesuai dengan struktur repository
RAW_DATA_PATH = "../data_raw/Telco-Customer-Churn.csv" 
OUTPUT_DIR = "namadataset_preprocessing"

def run_preprocessing():
    print("Memulai proses otomatisasi preprocessing dari file lokal...")
    
    # 1. Load Data dari folder data_raw
    if not os.path.exists(RAW_DATA_PATH):
        # Jika dijalankan dari root, sesuaikan path-nya
        df = pd.read_csv("data_raw/Telco-Customer-Churn.csv")
    else:
        df = pd.read_csv(RAW_DATA_PATH)
    
    # 2. Cleaning (Sesuai eksperimen notebook)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    df.drop(columns=['customerID'], inplace=True)
    
    # 3. Encoding
    le = LabelEncoder()
    df['Churn'] = le.fit_transform(df['Churn'])
    df = pd.get_dummies(df, drop_first=True)
    
    # 4. Splitting
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 5. Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Simpan sebagai DataFrame
    train_clean = pd.DataFrame(X_train_scaled, columns=X.columns)
    train_clean['Churn'] = y_train.values
    
    # 6. Simpan Output
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    train_clean.to_csv(f"{OUTPUT_DIR}/churn_clean.csv", index=False)
    print(f"Preprocessing Selesai! File disimpan di: {OUTPUT_DIR}/churn_clean.csv")

if __name__ == "__main__":
    run_preprocessing()