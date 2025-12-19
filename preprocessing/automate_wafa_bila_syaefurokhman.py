import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

def run_automation():
    # Load
    raw_path = 'dataset_raw/nearest-earth-objects(1910-2024).csv'
    if not os.path.exists(raw_path):
        print("Dataset Raw tidak ditemukan!")
        return
        
    df = pd.read_csv(raw_path)
    
    # Preprocessing
    df = df.drop(['neo_id', 'name', 'orbiting_body'], axis=1).dropna()
    df['is_hazardous'] = df['is_hazardous'].astype(int)
    
    X = df.drop('is_hazardous', axis=1)
    y = df['is_hazardous']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Save
    output_dir = 'preprocessing/nearest_earth_object_preprocessing'
    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame(X_train_scaled, columns=X.columns).to_csv(f'{output_dir}/X_train.csv', index=False)
    y_train.to_csv(f'{output_dir}/y_train.csv', index=False)
    print("Otomatisasi Berhasil!")

if __name__ == "__main__":
    run_automation()