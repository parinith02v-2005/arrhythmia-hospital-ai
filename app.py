import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import json
import torch.nn as nn
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
import tempfile

# -------- PAGE CONFIG -------- #
st.set_page_config(page_title="Mindsparks AI Hospital", layout="wide")

# -------- PREMIUM UI (GLASS EFFECT 🔥) -------- #
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg,#0f2027,#203a43,#2c5364);
}
.block-container {
    padding-top: 2rem;
}
.glass {
    background: rgba(255,255,255,0.08);
    border-radius: 15px;
    padding: 20px;
    backdrop-filter: blur(10px);
}
h1,h2,h3 {color:#00e6e6;}
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

model = ECGModel()

try:
    model.load_state_dict(torch.load("ecg_model.pth", map_location="cpu"))
    model.eval()
    model_loaded = True
except Exception as e:
    st.error(f"Model Error: {e}")
    model_loaded = False

# -------- BPM (NO SCIPY) -------- #
def calculate_bpm(signal):
    try:
        threshold = np.mean(signal)
        peaks = np.where(signal > threshold)[0]
        if len(peaks) > 1:
            rr = np.diff(peaks)
            bpm = 60 / (np.mean(rr) / 360)
            return int(bpm)
        return 0
    except:
        return 0

# -------- GRAD CAM -------- #
def grad_cam(model, signal_tensor):
    signal_tensor.requires_grad = True
    output = model(signal_tensor)
    class_idx = output.argmax()

    output[0, class_idx].backward()
    gradients = signal_tensor.grad[0][0]

    heatmap = gradients.abs().detach().numpy()
    return heatmap

# -------- PDF GENERATION -------- #
def generate_pdf(name, age, diagnosis, bpm):
    file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    doc = SimpleDocTemplate(file.name)
    styles = getSampleStyleSheet()

    content = [
        Paragraph(f"Patient: {name}", styles["Normal"]),
        Paragraph(f"Age: {age}", styles["Normal"]),
        Paragraph(f"Diagnosis: {diagnosis}", styles["Normal"]),
        Paragraph(f"BPM: {bpm}", styles["Normal"]),
    ]

    doc.build(content)
    return file.name

# -------- LOGIN -------- #
st.sidebar.title("🏥 Secure Login")
user = st.sidebar.text_input("Doctor ID")
password = st.sidebar.text_input("Password", type="password")

if user and password:

    st.title("🫀 Mindsparks AI ECG System")

    col1, col2 = st.columns([1,2])

    with col1:
        st.markdown("### Patient Info")
        name = st.text_input("Name")
        age = st.number_input("Age", 1, 120)
        spo2 = st.slider("SpO2", 70, 100)

        st.markdown("### Upload ECG")
        file = st.file_uploader("CSV / PNG / JPG", type=["csv","png","jpg"])

    if file and model_loaded:

        # -------- MULTI-LEAD SUPPORT -------- #
        if file.name.endswith(".csv"):
            signal = np.loadtxt(file, delimiter=",")
            if len(signal.shape) > 1:
                lead = st.selectbox("Select Lead", list(range(signal.shape[1])))
                signal = signal[:, lead]
        else:
            image = Image.open(file).convert("L")
            signal = np.array(image).flatten()

        # Normalize
        signal = (signal - np.mean(signal)) / np.std(signal)

        data = torch.tensor(signal[:200]).float().unsqueeze(0).unsqueeze(0)

        # -------- PREDICTION -------- #
        with torch.no_grad():
            output = model(data)
            pred = torch.argmax(output).item()
            confidence = torch.softmax(output, dim=1)[0][pred].item()

        bpm = calculate_bpm(signal)

        # -------- GRAD CAM -------- #
        heatmap = grad_cam(model, data)

        # -------- DISPLAY -------- #
        with col2:
            st.markdown("### ECG Signal + AI Highlight")

            fig, ax = plt.subplots()
            ax.plot(signal[:200], label="ECG")
            ax.plot(heatmap * max(signal[:200]), color='red', label="AI Focus")
            ax.legend()
            st.pyplot(fig)

            st.markdown("### Diagnosis")
            diagnosis = labels.get(str(pred), "Unknown")
            st.success(diagnosis)

            st.markdown("### Metrics")
            c1, c2, c3 = st.columns(3)
            c1.metric("BPM", bpm)
            c2.metric("SpO2", f"{spo2}%")
            c3.metric("Confidence", f"{confidence:.2f}")

            # -------- PDF -------- #
            pdf_path = generate_pdf(name, age, diagnosis, bpm)

            with open(pdf_path, "rb") as f:
                st.download_button("Download Medical Report", f, file_name="report.pdf")

            st.success("Analysis Complete")

else:
    st.warning("Login to continue")
