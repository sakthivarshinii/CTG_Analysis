# Fetal Health Classification

An AI-powered web application that classifies fetal health into three categories (Normal, Suspect, Pathological) based on Cardiotocography (CTG) data.

This project uses a Random Forest Classifier trained on the UCI Cardiotocography Dataset (ID 193) and serves predictions via a modern, glassmorphic Flask web dashboard.

## Features
- **Machine Learning**: Random Forest Classifier trained on key CTG features (Baseline FHR, Variability, Accelerations, Decelerations, Uterine Contractions).
- **Backend**: Flask API providing `/predict` and `/metrics` endpoints.
- **Frontend**: Clean, responsive, glassmorphic UI utilizing Vanilla HTML/CSS/JS and Chart.js.

## Requirements
- Python 3.8+
- The dependencies listed in `requirements.txt`

## How to Run Locally

1. **Clone the repository** (if applicable) and navigate to the project directory:
   ```bash
   cd ai_in_healthcare
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the Model** (This will fetch dataset, train the model, and create `model.pkl`):
   ```bash
   python model/train.py
   ```
   *Note: Ensure you have an active internet connection to download the UCI Dataset.*

4. **Start the Flask Backend**:
   ```bash
   python backend/app.py
   ```

5. **Access the Web Interface**:
   Open a web browser and navigate to:
   ```
   http://127.0.0.1:5000
   ```

## Usage
Enter the CTG values into the fields (e.g., FHR: 130, Variability: 75, etc.) and click "Analyze Data" to see the real-time prediction, confidence score, and interpretation message. The dashboard also displays visual charts for Model Metrics such as Confusion Matrix and Feature Importances.
