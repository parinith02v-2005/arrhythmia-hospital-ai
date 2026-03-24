import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import json
import torch.nn as nn

# -------- CONFIG -------- #
st.set_page_config(page_title="Clinical ECG AI", layout="wide")

# -------- CLEAN UI -------- #
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background:#0b1f2a;
}
h1,h2 {color:#00e6e6;}
.metric {font-size:22px;}
</style>
""", unsafe_allow_html=True)

# -------- LOAD LABELS -------- #
try:
    with open("labels.json") as f:
        labels = json.load(f)
except:
    labels = {"0":"Normal","1":"Atrial","2":"Ventricular","3":"Fusion","4":"Unknown"}

# -------- MODEL -------- #
class ECGModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1,32,7,padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32,64,5,padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64,128,3,padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(128*25,128),
            nn.ReLU(),
            nn.Linear(128,5)
        )

    def forward(self,x):
        x=self.conv(x)
        x=x.view(x.size(0),-1)
        return self.fc(x)

model = ECGModel()

try:
    model.load_state_dict(torch.load("ecg_model.pth", map_location="cpu"))
    model.eval()
    ok=True
except:
    ok=False

# -------- BPM FIX -------- #
def calculate_bpm(signal):
    peaks = []
    for i in range(1,len(signal)-1):
        if signal[i]>signal[i-1] and signal[i]>signal[i+1] and signal[i]>0.5:
            peaks.append(i)

    if len(peaks)>1:
        rr = np.diff(peaks)
        return int(60/(np.mean(rr)/360))
    return 0

# -------- HEATMAP -------- #
def grad_cam(model,x):
    x.requires_grad=True
    out=model(x)
    cls=out.argmax()
    out[0,cls].backward()
    return x.grad[0][0].abs().detach().numpy()

# -------- MAIN REGION -------- #
def get_region(heatmap, window=40):
    idx = np.argmax(heatmap)
    return max(0,idx-window), min(len(heatmap),idx+window)

# -------- LOGIN -------- #
st.sidebar.title("🏥 Doctor Login")
user=st.sidebar.text_input("ID")
pwd=st.sidebar.text_input("Password", type="password")

if user and pwd and ok:

    st.title("🫀 Clinical ECG AI Dashboard")

    col1,col2 = st.columns([1,2])

    with col1:
        name = st.text_input("Patient Name")
        age = st.number_input("Age",1,120)
        spo2 = st.slider("SpO2",70,100)
        file = st.file_uploader("Upload ECG", type=["csv","png","jpg"])

    if file:

        # LOAD SIGNAL
        if file.name.endswith(".csv"):
            signal = np.loadtxt(file, delimiter=",")
        else:
            img = Image.open(file).convert("L")
            signal = np.array(img).flatten()

        signal = (signal-np.mean(signal))/np.std(signal)

        x = torch.tensor(signal[:200]).float().unsqueeze(0).unsqueeze(0)

        # PREDICT
        out = model(x)
        pred = torch.argmax(out).item()
        conf = torch.softmax(out,dim=1)[0][pred].item()

        bpm = calculate_bpm(signal)

        heatmap = grad_cam(model,x)
        start,end = get_region(heatmap)

        # -------- GRAPH -------- #
        with col2:

            st.subheader("Full ECG Signal")

            fig, ax = plt.subplots(figsize=(12,4))

            # ECG GRID
            ax.set_facecolor("black")
            ax.grid(True, color='gray', linestyle='--', linewidth=0.3)

            ax.plot(signal, color='lime', linewidth=1.5)

            # SINGLE REGION
            ax.axvspan(start, end, color='green', alpha=0.3)

            st.pyplot(fig)

            diagnosis = labels.get(str(pred),"Unknown")

            # -------- CLEAN METRICS -------- #
            c1,c2,c3 = st.columns(3)
            c1.metric("Heart Rate (BPM)", bpm)
            c2.metric("SpO2", f"{spo2}%")
            c3.metric("Confidence", f"{conf:.2f}")

            st.success(f"Diagnosis: {diagnosis}")

            st.markdown("### Clinical Summary")
            st.info(f"""
            Rhythm: {"Irregular" if bpm<60 or bpm>100 else "Normal"}  
            Abnormal segment localized using AI  
            Model confidence: {conf:.2f}  
            """)

else:
    st.warning("Login required")
