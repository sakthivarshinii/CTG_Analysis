import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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

    # Select required features mapping exactly to user request:
    # Baseline FHR -> 'LB'
    # Variability -> 'ASTV'
    # Accelerations -> 'AC'
    # Decelerations -> 'DL'
    # Uterine Contractions -> 'UC'
    features_to_use = ['LB', 'ASTV', 'AC', 'DL', 'UC']
    
    # Check if features exist
    if not all(col in X.columns for col in features_to_use):
        print(f"Error: Not all required features are present in the dataset. Available: {list(X.columns)}")
        return

    X_selected = X[features_to_use].copy()
    
    # Fill any NaNs
    X_selected.fillna(X_selected.mean(), inplace=True)
    
    target_col = 'NSP'
    y_target = y[target_col].copy()

    # Drop NaNs from target and align X
    valid_idx = y_target.dropna().index
    X_selected = X_selected.loc[valid_idx]
    y_target = y_target.loc[valid_idx]

    print("Data loaded. Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y_target, test_size=0.2, random_state=42, stratify=y_target
    )

    print("Training Random Forest Classifier on 5 simplified features...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    print("Evaluating model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Accuracy: {accuracy * 100:.2f}%")

    importances = model.feature_importances_
    feature_importance_dict = {feat: float(imp) for feat, imp in zip(features_to_use, importances)}

    metrics = {
        "accuracy": float(accuracy),
        "confusion_matrix": cm.tolist(),
        "classes": [1, 2, 3], # 1: Normal, 2: Suspect, 3: Pathological
        "feature_importance": feature_importance_dict
    }

    os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)
    
    out_dir = os.path.dirname(__file__)
    
    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
        
    with open(os.path.join(out_dir, 'model.pkl'), 'wb') as f:
        pickle.dump(model, f)

    print("Model ('model.pkl') and metrics ('metrics.json') saved successfully in the 'model' directory.")

if __name__ == '__main__':
    main()
