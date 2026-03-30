import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="NeuroFetus AI", page_icon="📈", layout="wide")

# Connect to the FastAPI backend running in the background
API_URL = "http://127.0.0.1:8000"

st.title("NeuroFetus AI Clinical Support")

# Authentication state
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = ""

if not st.session_state.authenticated:
    st.subheader("Login")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Log In")
        
        if submit:
            res = requests.post(f"{API_URL}/api/login", json={"username": username, "password": password})
            if res.status_code == 200:
                st.session_state.authenticated = True
                st.session_state.username = username
                st.rerun()
            else:
                st.error("Invalid credentials (try checking database setup or default user/pass).")
else:
    # Sidebar
    st.sidebar.title(f"Welcome, Dr. {st.session_state.username}")
    if st.sidebar.button("Logout"):
        st.session_state.authenticated = False
        st.rerun()
        
    st.sidebar.header("Recent Analyses")
    try:
        hist_res = requests.get(f"{API_URL}/api/history?limit=10")
        if hist_res.status_code == 200:
            hist_data = hist_res.json()
            if hist_data:
                df = pd.DataFrame(hist_data)
                df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime("%H:%M %m/%d")
                df = df[['patient_id', 'prediction', 'timestamp']]
                st.sidebar.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.sidebar.info("No recent analyses")
    except Exception as e:
        st.sidebar.error("Could not fetch history. Is FastAPI running?")

    # Main dashboard
    col1, col2 = st.columns([1.2, 1])
    
    with col1:
        st.subheader("Patient Data Input")
        with st.form("patient_form"):
            patient_id = st.text_input("Patient ID", "PT-1002")
            lb = st.slider("Baseline FHR (LB)", 50, 250, 133)
            astv = st.slider("Abnormal STV (%)", 0, 100, 73)
            
            subcol1, subcol2, subcol3 = st.columns(3)
            ac = subcol1.number_input("Accelerations (AC)", min_value=0.0, value=0.000, step=0.001, format="%.3f")
            dl = subcol2.number_input("Decelerations (DL)", min_value=0.0, value=0.003, step=0.001, format="%.3f")
            uc = subcol3.number_input("Contractions (UC)", min_value=0.0, value=0.005, step=0.001, format="%.3f")
            
            analyze = st.form_submit_button("Run AI Analysis", type="primary")
            
        if analyze:
            payload = {
                "patient_id": patient_id, "LB": lb, "ASTV": astv, "AC": ac, "DL": dl, "UC": uc
            }
            try:
                with st.spinner('Analyzing metrics...'):
                    res = requests.post(f"{API_URL}/api/predict", json=payload)
                if res.status_code == 200:
                    st.session_state.last_prediction = res.json()
                    st.session_state.last_features = payload
                else:
                    st.error(f"Error ({res.status_code}): {res.text}")
            except Exception as e:
                st.error("Failed to connect to backend API engine. Please ensure 'uvicorn backend.app:app' is running.")


    with col2:
        st.subheader("AI Assessment")
        
        if "last_prediction" in st.session_state:
            pred_data = st.session_state.last_prediction
            
            is_normal = pred_data["prediction"] == "Normal"
            st.metric(
                label=f"Risk: {pred_data['risk_level']}", 
                value=pred_data["prediction"], 
                delta=f"Confidence: {pred_data['confidence']}", 
                delta_color="normal" if is_normal else "inverse"
            )
            
            st.write(f"**Ensemble Agreement:** {pred_data['agreement']}")
            
            if is_normal:
                st.success(f"**Explanation:** {pred_data['explanation']}")
            else:
                st.warning(f"**Explanation:** {pred_data['explanation']}")
            
            # SHAP Visualization
            st.write("### Decision Drivers")
            shap_df = pd.DataFrame.from_dict(pred_data["top_features"], orient="index", columns=["Impact Score"])
            st.bar_chart(shap_df, horizontal=True)
            
        else:
            st.info("Awaiting Data. Please submit the form to generate an AI assessment.")
            
    st.divider()
    
    # AI CHAT
    st.subheader("Clinical AI Assistant")
    
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm analyzing the CTG metrics. How can I assist you today?"}
        ]
        
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    col_chip1, col_chip2 = st.columns(2)
    with col_chip1:
        if st.button("Explain abnormal values"):
            prompt = "Explain abnormal values"
            st.session_state.pending_prompt = prompt
    with col_chip2:
        if st.button("Summarize patient condition"):
            prompt = "Summarize patient condition"
            st.session_state.pending_prompt = prompt            
            
    prompt = st.chat_input("Ask about the diagnosis...")
    
    if "pending_prompt" in st.session_state and st.session_state.pending_prompt:
        prompt = st.session_state.pending_prompt
        st.session_state.pending_prompt = None

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            if "last_prediction" in st.session_state:
                chat_payload = {
                    "message": prompt,
                    "prediction": st.session_state.last_prediction["prediction"],
                    "features": st.session_state.last_features
                }
            else:
                chat_payload = {
                    "message": prompt,
                    "prediction": "Unknown",
                    "features": {}
                }
                
            try:
                chat_res = requests.post(f"{API_URL}/api/chat", json=chat_payload)
                if chat_res.status_code == 200:
                    ai_resp = chat_res.json()["response"]
                    st.markdown(ai_resp)
                    st.session_state.messages.append({"role": "assistant", "content": ai_resp})
                else:
                    st.error("Chat API error.")
            except:
                st.error("Failed to reach Chat engine.")
