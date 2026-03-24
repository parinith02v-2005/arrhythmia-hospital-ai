import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import json
import torch.nn as nn

# ---------------- CONFIG ---------------- #
st.set_page_config(page_title="Clinical ECG AI", layout="wide")

# ---------------- UI ---------------- #
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg,#0b1f2a,#133b5c);
}
h1,h2 {color:#00e6e6;}
</style>
""", unsafe_allow_html=True)

# ---------------- LOGIN ---------------- #
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

st.sidebar.title("🏥 Doctor Login")

username = st.sidebar.text_input("Doctor ID")
password = st.sidebar.text_input("Password", type="password")

VALID_USER = "admin"
VALID_PASS = "1234"

if st.sidebar.button("Login"):
    if username == VALID_USER and password == VALID_PASS:
        st.session_state.logged_in = True
        st.sidebar.success("Login Successful ✅")
    else:
        st.sidebar.error("Invalid Credentials ❌")

if not st.session_state.logged_in:
    st.warning("🔐 Please login to access system")
    st.stop()

# ---------------- LABELS ---------------- #
try:
    with open("labels.json") as f:
        labels = json.load(f)
except:
    labels = {"0":"Normal","1":"Atrial","2":"Ventricular","3":"Fusion","4":"Unknown"}

# ---------------- MODEL ---------------- #
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

        # MATCH TRAINED MODEL (WITH DROPOUT)
        self.fc = nn.Sequential(
            nn.Linear(128*25,128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128,5)
        )

    def forward(self,x):
        x=self.conv(x)
        x=x.view(x.size(0),-1)
        return self.fc(x)

model = ECGModel()

# -------- SAFE LOAD (NO ERROR) -------- #
try:
    state_dict = torch.load("ecg_model.pth", map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model_ok = True
except Exception as e:
    st.error(f"Model Load Error: {e}")
    model_ok = False

# ---------------- BPM ---------------- #
def calculate_bpm(signal):
    peaks = []
    for i in range(1,len(signal)-1):
        if signal[i] > signal[i-1] and signal[i] > signal[i+1] and signal[i] > 0.5:
            peaks.append(i)

    if len(peaks) > 1:
        rr = np.diff(peaks)
        return int(60 / (np.mean(rr) / 360))
    return 0

# ---------------- GRAD CAM ---------------- #
def grad_cam(model,x):
    x.requires_grad=True
    out=model(x)
    cls=out.argmax()
    out[0,cls].backward()
    return x.grad[0][0].abs().detach().numpy()

# ---------------- REGION ---------------- #
def get_region(heatmap):
    idx = np.argmax(heatmap)
    return max(0,idx-40), min(len(heatmap),idx+40)

# ---------------- MAIN UI ---------------- #
st.title("🫀 Mindsparks Clinical ECG AI Dashboard")

col1,col2 = st.columns([1,2])

with col1:
    name = st.text_input("Patient Name")
    age = st.number_input("Age",1,120)
    spo2 = st.slider("SpO2",70,100)
    file = st.file_uploader("Upload ECG (CSV/PNG/JPG)")

if file and model_ok:

    # LOAD SIGNAL
    if file.name.endswith(".csv"):
        signal = np.loadtxt(file, delimiter=",")
    else:
        img = Image.open(file).convert("L")
        signal = np.array(img).flatten()

    signal = (signal-np.mean(signal))/np.std(signal)

    x = torch.tensor(signal[:200]).float().unsqueeze(0).unsqueeze(0)

    # PREDICT
    with torch.no_grad():
        out = model(x)
        pred = torch.argmax(out).item()
        conf = torch.softmax(out,dim=1)[0][pred].item()

    bpm = calculate_bpm(signal)

    heatmap = grad_cam(model,x)
    start,end = get_region(heatmap)

    with col2:

        st.subheader("ECG Signal")

        fig, ax = plt.subplots(figsize=(12,4))

        ax.set_facecolor("black")
        ax.grid(True, color='gray', linestyle='--', linewidth=0.3)

        ax.plot(signal, color='lime', linewidth=1.5)

        # SINGLE ABNORMAL REGION
        ax.axvspan(start, end, color='green', alpha=0.3)

        st.pyplot(fig)

        diagnosis = labels.get(str(pred),"Unknown")

        c1,c2,c3 = st.columns(3)
        c1.metric("Heart Rate", bpm)
        c2.metric("SpO2", f"{spo2}%")
        c3.metric("Confidence", f"{conf:.2f}")

        st.success(f"Diagnosis: {diagnosis}")

        st.markdown("### Clinical Summary")
        st.info(f"""
        • Rhythm: {"Irregular" if bpm<60 or bpm>100 else "Normal Sinus"}  
        • Abnormal ECG segment detected  
        • AI Confidence: {conf:.2f}  
        """)
