import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from ucimlrepo import fetch_ucirepo
import pickle
import json
import os

def main():
    print("Fetching dataset Cardiotocography (ID 193) via ucimlrepo...")
    try:
        cardiotocography = fetch_ucirepo(id=193) 
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return
        
    X = cardiotocography.data.features 
    y = cardiotocography.data.targets 

    features_to_use = ['LB', 'ASTV', 'AC', 'DL', 'UC']
    
    if not all(col in X.columns for col in features_to_use):
        print(f"Error: Missing features. Available: {list(X.columns)}")
        return

    X_selected = X[features_to_use].copy()
    X_selected.fillna(X_selected.mean(), inplace=True)
    
    target_col = 'NSP'
    y_target = y[target_col].copy()

    valid_idx = y_target.dropna().index
    X_selected = X_selected.loc[valid_idx]
    y_target = y_target.loc[valid_idx]

    # Target classes: 0: Normal, 1: Suspect, 2: Pathological
    y_target = y_target - 1 

    print("Scaling and splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y_target, test_size=0.2, random_state=42, stratify=y_target
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    metrics = {}

    # --- 1. Random Forest ---
    print("Training Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    rf_pred = rf_model.predict(X_test_scaled)
    metrics['RandomForest'] = {
        'accuracy': float(accuracy_score(y_test, rf_pred)),
        'confusion_matrix': confusion_matrix(y_test, rf_pred).tolist()
    }

    # --- 2. XGBoost ---
    print("Training XGBoost...")
    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    xgb_model.fit(X_train_scaled, y_train)
    xgb_pred = xgb_model.predict(X_test_scaled)
    metrics['XGBoost'] = {
        'accuracy': float(accuracy_score(y_test, xgb_pred)),
        'confusion_matrix': confusion_matrix(y_test, xgb_pred).tolist()
    }

    # --- 3. ANN (MLPClassifier) ---
    print("Training Neural Network (MLP)...")
    ann_model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    ann_model.fit(X_train_scaled, y_train)
    ann_pred = ann_model.predict(X_test_scaled)
    metrics['ANN'] = {
        'accuracy': float(accuracy_score(y_test, ann_pred)),
        'confusion_matrix': confusion_matrix(y_test, ann_pred).tolist()
    }

    print("Model evaluations:", {k: v['accuracy'] for k, v in metrics.items()})

    out_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, 'rf_model.pkl'), 'wb') as f:
        pickle.dump(rf_model, f)
    
    with open(os.path.join(out_dir, 'xgb_model.pkl'), 'wb') as f:
        pickle.dump(xgb_model, f)
        
    with open(os.path.join(out_dir, 'ann_model.pkl'), 'wb') as f:
        pickle.dump(ann_model, f)
    
    with open(os.path.join(out_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)

    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

    print("Models, scaler, and metrics saved successfully to 'models'.")

if __name__ == '__main__':
    main()
