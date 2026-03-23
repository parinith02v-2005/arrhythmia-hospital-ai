import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import json
import torch.nn as nn

# -------- PAGE CONFIG -------- #
st.set_page_config(page_title="Mindsparks Hospital AI", layout="wide")

# -------- CUSTOM CSS (NEXT LEVEL UI 🔥) -------- #
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
.card {
    padding: 20px;
    border-radius: 15px;
    background: rgba(255,255,255,0.05);
    box-shadow: 0px 0px 10px rgba(0,255,255,0.3);
}
</style>
""", unsafe_allow_html=True)

# -------- LOAD LABELS -------- #
with open("labels.json") as f:
    labels = json.load(f)

# -------- MODEL -------- #
class ECGModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, 7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(64*47, 128),
            nn.ReLU(),
            nn.Linear(128, 5)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

model = ECGModel()
model.load_state_dict(torch.load("ecg_model.pth", map_location="cpu"))
model.eval()

# -------- LOGIN SYSTEM 🔐 -------- #
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
        file = st.file_uploader("Drag & Drop ECG File", type=["csv","png","jpg"])

    if file:
        if file.name.endswith(".csv"):
            signal = np.loadtxt(file, delimiter=",")
        else:
            image = Image.open(file).convert("L")
            signal = np.array(image).flatten()

        signal = (signal - np.mean(signal)) / np.std(signal)

        data = torch.tensor(signal[:200]).float().unsqueeze(0).unsqueeze(0)

        output = model(data)
        pred = torch.argmax(output).item()

        confidence = torch.softmax(output, dim=1)[0][pred].item()

        with col2:
            st.markdown("### 📊 ECG Signal")
            st.line_chart(signal[:1000])

            st.markdown("### 🧠 Diagnosis")
            st.success(labels[str(pred)])

            st.markdown("### 📈 Metrics")
            c1, c2, c3 = st.columns(3)

            c1.metric("💓 BPM", "72")
            c2.metric("🫁 SpO2", f"{spo2}%")
            c3.metric("📊 Confidence", f"{confidence:.2f}")

            st.markdown("### 📋 Report")
            st.info(f"""
            Patient: {name}  
            Age: {age}  
            Diagnosis: {labels[str(pred)]}  
            Confidence: {confidence:.2f}
            """)

            st.balloons()

else:
    st.warning("🔐 Please login to access the system")
