from fastapi import FastAPI, Depends, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session
from typing import List
import pickle
import json
import os
import numpy as np
import shap

# sys path for imports if needed
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.db import SessionLocal, PatientRecord, init_db

app = FastAPI(title="Fetal Health Decision Support API")

# Mount frontend
app.mount("/static", StaticFiles(directory="frontend"), name="static")

# Load models
base_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(base_dir, '../models')

rf_model = None
xgb_model = None
ann_model = None
scaler = None
explainer = None

def load_ml_models():
    global rf_model, xgb_model, ann_model, scaler, explainer
    try:
        with open(os.path.join(models_dir, 'rf_model.pkl'), 'rb') as f: rf_model = pickle.load(f)
        with open(os.path.join(models_dir, 'xgb_model.pkl'), 'rb') as f: xgb_model = pickle.load(f)
        with open(os.path.join(models_dir, 'ann_model.pkl'), 'rb') as f: ann_model = pickle.load(f)
        with open(os.path.join(models_dir, 'scaler.pkl'), 'rb') as f: scaler = pickle.load(f)
        
        # Initialize SHAP explainer from Random Forest
        explainer = shap.TreeExplainer(rf_model)
        print("Models and SHAP Explainer loaded successfully.")
    except Exception as e:
        print(f"Warning: Could not load models. Did you run models/train.py? Error: {e}")

load_ml_models()
init_db()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Schema
class CTGInput(BaseModel):
    patient_id: str = "Unknown"
    LB: float
    ASTV: float
    AC: float
    DL: float
    UC: float

# Root
@app.get("/")
def read_index():
    return FileResponse('frontend/index.html')

@app.post("/api/predict")
def predict(data: CTGInput, db: Session = Depends(get_db)):
    if not rf_model or not xgb_model or not ann_model:
        raise HTTPException(status_code=500, detail="Models not loaded. Please train models first.")
        
    features = ['LB', 'ASTV', 'AC', 'DL', 'UC']
    input_list = [data.LB, data.ASTV, data.AC, data.DL, data.UC]
    input_array = np.array(input_list).reshape(1, -1)
    
    # Scale
    X_scaled = scaler.transform(input_array)
    
    # Predictions
    rf_pred = int(rf_model.predict(X_scaled)[0])
    xgb_pred = int(xgb_model.predict(X_scaled)[0])
    ann_pred = int(ann_model.predict(X_scaled)[0])
    
    # Confidence from models
    rf_proba = max(rf_model.predict_proba(X_scaled)[0]) * 100
    xgb_proba = max(xgb_model.predict_proba(X_scaled)[0]) * 100
    ann_proba = max(ann_model.predict_proba(X_scaled)[0]) * 100
    
    # Voting ensemble
    votes = [rf_pred, xgb_pred, ann_pred]
    final_pred = int(max(set(votes), key=votes.count))
    agreement = votes.count(final_pred)
    
    # Classes: 0: Normal, 1: Suspect, 2: Pathological
    class_map = {0: "Normal", 1: "Suspect", 2: "Pathological"}
    risk_map = {0: "Stable condition", 1: "Monitor closely", 2: "High risk - immediate medical attention required"}
    
    final_label = class_map.get(final_pred, "Unknown")
    risk_level = risk_map.get(final_pred, "Unknown")
    
    # Average confidence among those who voted for final
    confidences = []
    if rf_pred == final_pred: confidences.append(rf_proba)
    if xgb_pred == final_pred: confidences.append(xgb_proba)
    if ann_pred == final_pred: confidences.append(ann_proba)
    final_confidence = np.mean(confidences) if confidences else 0
    
    # SHAP explainer
    shap_values = explainer.shap_values(X_scaled)
    # Check if shap_values is a list (multi-class tree) or single array
    if isinstance(shap_values, list):
        target_class_shap = shap_values[int(final_pred)][0]
    elif len(shap_values.shape) == 3: # Some versions of shap return 3D array (n_samples, n_features, n_classes)
        target_class_shap = shap_values[0, :, int(final_pred)]
    else:
        target_class_shap = shap_values[0]
        
    feature_abs_shap = np.abs(target_class_shap)
    top_feature_idx = np.argmax(feature_abs_shap)
    top_feature = features[top_feature_idx]
    
    explanation = f"Top contributing feature: {top_feature}. "
    if top_feature == 'DL' and input_list[3] > 0.005: 
        explanation += "High decelerations indicating potential distress."
    elif top_feature == 'ASTV' and input_list[1] > 60:
        explanation += "High abnormal short term variability."
    elif top_feature == 'LB':
        explanation += f"Baseline Fetal Heart Rate is {input_list[0]}."
    else:
        explanation += "Model based on complex interaction of CTG readings."

    # Save to db
    record = PatientRecord(
        patient_id=data.patient_id,
        LB=data.LB, ASTV=data.ASTV, AC=data.AC, DL=data.DL, UC=data.UC,
        prediction=final_label,
        confidence=final_confidence,
        risk_level=risk_level
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    return {
        "prediction": final_label,
        "confidence": f"{final_confidence:.1f}%",
        "agreement": f"{agreement}/3",
        "risk_level": risk_level,
        "explanation": explanation,
        "top_features": {f: float(v) for f, v in zip(features, target_class_shap)}
    }

@app.get("/api/history")
def get_history(limit: int = 10, db: Session = Depends(get_db)):
    records = db.query(PatientRecord).order_by(PatientRecord.timestamp.desc()).limit(limit).all()
    return records

class ChatInput(BaseModel):
    message: str
    prediction: str
    features: dict

@app.post("/api/chat")
def chat(data: ChatInput):
    msg = data.message.lower()
    if "why" in msg and "pathological" in data.prediction.lower():
        return {"response": f"The prediction is Pathological because features (especially FHR Variability and Decelerations) indicate high fetal distress patterns in our ML ensemble. Your values: {data.features}"}
    elif "normal" in data.prediction.lower():
        return {"response": "The model sees normal ranges for Fetal Heart Rate and Variability, similar to healthy baseline data."}
    else:
        return {"response": "I am an AI assistant. I look at CTG data (LB, ASTV, AC, DL, UC) and use Random Forest, XGBoost and Neural Networks to classify fetal health."}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
