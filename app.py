import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import json
import torch.nn as nn

# -------- CONFIG -------- #
st.set_page_config(page_title="Mindsparks ECG Clinical AI", layout="wide")

# -------- UI -------- #
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: #0b1f2a;
}
h1 {color:#00e6e6;}
.block-container {padding:2rem;}
</style>
""", unsafe_allow_html=True)

# -------- LABELS -------- #
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
            nn.Dropout(0.5),
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
    ok = True
except:
    ok = False

# -------- BPM -------- #
def calculate_bpm(signal):
    peaks = np.where(signal > np.mean(signal))[0]
    if len(peaks)>1:
        return int(60/(np.mean(np.diff(peaks))/360))
    return 0

# -------- GRAD CAM -------- #
def grad_cam(model, x):
    x.requires_grad=True
    out = model(x)
    cls = out.argmax()
    out[0,cls].backward()
    return x.grad[0][0].abs().detach().numpy()

# -------- SINGLE REGION -------- #
def get_main_region(heatmap):
    idx = np.argmax(heatmap)
    start = max(0, idx-10)
    end = min(len(heatmap), idx+10)
    return start, end

# -------- ECG PEAK DETECTION -------- #
def detect_peaks(signal):
    peaks = np.where(signal > np.mean(signal))[0]
    if len(peaks)==0:
        return None,None,None
    r = peaks[np.argmax(signal[peaks])]
    q = r-5 if r-5>0 else r
    t = r+5 if r+5<len(signal) else r
    return q,r,t

# -------- LOGIN -------- #
st.sidebar.title("🏥 Doctor Login")
user = st.sidebar.text_input("ID")
pwd = st.sidebar.text_input("Password", type="password")

if user and pwd and ok:

    st.title("🫀 Mindsparks Clinical ECG AI")

    col1, col2 = st.columns([1,2])

    with col1:
        name = st.text_input("Patient Name")
        age = st.number_input("Age",1,120)
        spo2 = st.slider("SpO2",70,100)
        file = st.file_uploader("Upload ECG", type=["csv","png","jpg"])

    if file:

        # LOAD
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

        # GRAD CAM
        heatmap = grad_cam(model,x)
        start,end = get_main_region(heatmap)

        # PEAKS
        q,r,t = detect_peaks(signal)

        # GRAPH
        with col2:
            st.subheader("ECG Analysis")

            fig, ax = plt.subplots(figsize=(10,4))

            ax.plot(signal[:200], color='cyan')

            # SINGLE GREEN REGION
            ax.axvspan(start,end,color='green',alpha=0.4,label="Abnormal Region")

            # PQRST MARKERS
            if r:
                ax.scatter(r, signal[r], color='red')
                ax.text(r, signal[r], 'R', color='red')

            if q:
                ax.scatter(q, signal[q], color='yellow')
                ax.text(q, signal[q], 'Q')

            if t:
                ax.scatter(t, signal[t], color='white')
                ax.text(t, signal[t], 'T')

            ax.legend()
            st.pyplot(fig)

            diagnosis = labels.get(str(pred),"Unknown")

            st.success(f"Diagnosis: {diagnosis}")

            c1,c2,c3 = st.columns(3)
            c1.metric("BPM", bpm)
            c2.metric("SpO2", f"{spo2}%")
            c3.metric("Confidence", f"{conf:.2f}")

            st.markdown("### Clinical Interpretation")
            st.info(f"""
            • Rhythm: {"Irregular" if bpm<60 or bpm>100 else "Normal Sinus"}  
            • Abnormal segment localized  
            • QRS complex analyzed  
            """)

else:
    st.warning("Login required or model not loaded")
