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

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(page_title="Mindsparks Clinical ECG AI", layout="wide")

# ---------------- PREMIUM UI ---------------- #
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg,#0b1f2a,#133b5c);
}
.card {
    background: rgba(255,255,255,0.05);
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.3);
}
h1,h2,h3 {color:#00e6e6;}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD LABELS ---------------- #
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
    st.error(f"Model load error: {e}")
    model_loaded = False

# ---------------- BPM ---------------- #
def calculate_bpm(signal):
    threshold = np.mean(signal)
    peaks = np.where(signal > threshold)[0]
    if len(peaks) > 1:
        rr = np.diff(peaks)
        return int(60 / (np.mean(rr) / 360))
    return 0

# ---------------- GRAD CAM ---------------- #
def grad_cam(model, signal_tensor):
    signal_tensor.requires_grad = True
    output = model(signal_tensor)
    class_idx = output.argmax()
    output[0, class_idx].backward()
    gradients = signal_tensor.grad[0][0]
    return gradients.abs().detach().numpy()

# ---------------- HIGHLIGHT ---------------- #
def highlight_abnormal(heatmap):
    threshold = np.percentile(heatmap, 85)
    return heatmap > threshold

# ---------------- PDF ---------------- #
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

# ---------------- LOGIN ---------------- #
st.sidebar.title("🏥 Doctor Login")
user = st.sidebar.text_input("Doctor ID")
password = st.sidebar.text_input("Password", type="password")

if user and password:

    st.title("🫀 Mindsparks Clinical ECG AI System")

    col1, col2 = st.columns([1,2])

    with col1:
        st.markdown("### Patient Details")
        name = st.text_input("Name")
        age = st.number_input("Age", 1, 120)
        spo2 = st.slider("SpO2", 70, 100)

        file = st.file_uploader("Upload ECG", type=["csv","png","jpg"])

    if file and model_loaded:

        # -------- LOAD SIGNAL -------- #
        if file.name.endswith(".csv"):
            signal = np.loadtxt(file, delimiter=",")
            if len(signal.shape) > 1:
                lead = st.selectbox("Select Lead", list(range(signal.shape[1])))
                signal = signal[:, lead]
        else:
            image = Image.open(file).convert("L")
            signal = np.array(image).flatten()

        signal = (signal - np.mean(signal)) / np.std(signal)

        data = torch.tensor(signal[:200]).float().unsqueeze(0).unsqueeze(0)

        # -------- PREDICT -------- #
        with torch.no_grad():
            output = model(data)
            pred = torch.argmax(output).item()
            confidence = torch.softmax(output, dim=1)[0][pred].item()

        bpm = calculate_bpm(signal)

        # -------- EXPLANATION -------- #
        heatmap = grad_cam(model, data)
        abnormal = highlight_abnormal(heatmap[:200])

        # -------- GRAPH -------- #
        with col2:
            st.markdown("### ECG with AI Highlight")

            fig, ax = plt.subplots(figsize=(10,4))
            ax.plot(signal[:200], color='cyan', linewidth=2)

            for i in range(len(abnormal)):
                if abnormal[i]:
                    ax.axvspan(i, i+1, color='green', alpha=0.3)

            st.pyplot(fig)

            # -------- RESULTS -------- #
            diagnosis = labels.get(str(pred), "Unknown")

            st.success(f"Diagnosis: {diagnosis}")

            c1, c2, c3 = st.columns(3)
            c1.metric("BPM", bpm)
            c2.metric("SpO2", f"{spo2}%")
            c3.metric("Confidence", f"{confidence:.2f}")

            # -------- CLINICAL SUMMARY -------- #
            st.markdown("### Clinical Summary")
            st.info(f"""
            Diagnosis: {diagnosis}  
            Rhythm: {"Irregular" if bpm < 60 or bpm > 100 else "Normal Sinus"}  
            AI Confidence: {confidence:.2f}  
            """)

            # -------- AI EXPLANATION -------- #
            st.markdown("### AI Explanation")
            st.info("""
            Green regions indicate abnormal ECG segments detected by AI.
            """)

            # -------- PDF -------- #
            pdf = generate_pdf(name, age, diagnosis, bpm)
            with open(pdf, "rb") as f:
                st.download_button("Download Report", f, file_name="ECG_Report.pdf")

else:
    st.warning("Login required")
