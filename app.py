import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import json
import torch.nn as nn
from scipy.signal import find_peaks

# -------- PAGE CONFIG -------- #
st.set_page_config(page_title="Mindsparks Hospital AI", layout="wide")

# -------- CUSTOM UI -------- #
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
}
h1, h2, h3 {
    color: #00e6e6;
}
.stButton>button {
    background: #00e6e6;
    border-radius: 12px;
    height: 3em;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# -------- LOAD LABELS -------- #
try:
    with open("labels.json") as f:
        labels = json.load(f)
except:
    labels = {"0":"Normal","1":"Atrial","2":"Ventricular","3":"Fusion","4":"Unknown"}

# -------- MODEL (MATCHED WITH TRAINING) -------- #
class ECGModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, 7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, 5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        self.fc = nn.Sequential(
            nn.Linear(128*25, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 5)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# -------- LOAD MODEL SAFELY -------- #
model = ECGModel()

try:
    state_dict = torch.load("ecg_model.pth", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    model_loaded = True
except Exception as e:
    st.error(f"Model loading failed: {e}")
    model_loaded = False

# -------- BPM FUNCTION -------- #
def calculate_bpm(signal, fs=360):
    try:
        peaks, _ = find_peaks(signal, distance=fs*0.6)
        if len(peaks) > 1:
            rr = np.diff(peaks) / fs
            return int(60 / np.mean(rr))
        return 0
    except:
        return 0

# -------- LOGIN -------- #
st.sidebar.title("🏥 Doctor Login")
user = st.sidebar.text_input("Doctor ID")
password = st.sidebar.text_input("Password", type="password")

if user and password:

    st.title("🫀 Mindsparks ECG AI Dashboard")

    col1, col2 = st.columns([1,2])

    with col1:
        st.markdown("### 🧍 Patient Details")
        name = st.text_input("Name")
        age = st.number_input("Age", 1, 120)
        spo2 = st.slider("SpO2", 70, 100)

        st.markdown("### 📂 Upload ECG")
        file = st.file_uploader("Upload ECG (CSV / PNG / JPG)", type=["csv","png","jpg"])

    if file and model_loaded:

        # -------- LOAD SIGNAL -------- #
        try:
            if file.name.endswith(".csv"):
                signal = np.loadtxt(file, delimiter=",")
            else:
                image = Image.open(file).convert("L")
                signal = np.array(image).flatten()

            # Normalize
            signal = (signal - np.mean(signal)) / np.std(signal)

        except:
            st.error("Error reading ECG file")
            st.stop()

        # -------- MODEL INPUT -------- #
        data = torch.tensor(signal[:200]).float().unsqueeze(0).unsqueeze(0)

        # -------- PREDICTION -------- #
        with torch.no_grad():
            output = model(data)
            pred = torch.argmax(output).item()
            confidence = torch.softmax(output, dim=1)[0][pred].item()

        bpm = calculate_bpm(signal)

        # -------- DISPLAY -------- #
        with col2:
            st.markdown("### 📊 ECG Signal")
            st.line_chart(signal[:1000])

            st.markdown("### 🧠 Diagnosis")
            st.success(labels.get(str(pred), "Unknown"))

            st.markdown("### 📈 Metrics")
            c1, c2, c3 = st.columns(3)

            c1.metric("💓 BPM", f"{bpm}")
            c2.metric("🫁 SpO2", f"{spo2}%")
            c3.metric("📊 Confidence", f"{confidence:.2f}")

            st.markdown("### 📋 Medical Report")
            st.info(f"""
            Patient Name: {name}  
            Age: {age}  
            Diagnosis: {labels.get(str(pred), "Unknown")}  
            Confidence: {confidence:.2f}  
            Heart Rate: {bpm} BPM  
            """)

            st.success("✅ AI Analysis Complete")

else:
    st.warning("🔐 Please login to access the system")
