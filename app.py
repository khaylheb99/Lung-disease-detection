import streamlit as st
import torch
import torch.nn as nn
from torchvision.models import swin_t, Swin_T_Weights
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
import requests


st.title("**Lung Disease Detection using Swin Transformer**")
st.write("Upload a chest CT scan or X-ray to predict the type of lung disease using a fine-tuned **Swin Transformer** model.")

class_names = [
    "Normal",
    "Large Cell Carcinoma",
    "Squamous Cell Carcinoma",
    "Adenocarcinoma",
    "Benign"
]

MODEL_PATH = "model.pth"
DRIVE_URL = "https://drive.google.com/file/d/1SWOehqN5jmJW0t90b9llUxngrhhlCfNT/view?usp=sharing"


from torchvision.models import swin_t, Swin_T_Weights
import torch
import os
import requests
import streamlit as st

class_names = ["Normal", "Large Cell Carcinoma", "Squamous Cell Carcinoma", "Adenocarcinoma", "Benign"]
MODEL_PATH = "swin_lung_model.pth"
DRIVE_URL = "https://drive.google.com/uc?id=YOUR_FILE_ID"  # 👈 replace with actual file ID


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model from Google Drive... (only once)"):
            r = requests.get(DRIVE_URL)
            open(MODEL_PATH, "wb").write(r.content)
            st.success("Model downloaded successfully!")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weights = Swin_T_Weights.DEFAULT
    model = swin_t(weights=None)
    model.head = torch.nn.Linear(model.head.in_features, len(class_names))

    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()
    return model


model = load_model()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# @st.cache_resource
# def load_model():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     weights = Swin_T_Weights.DEFAULT
#     model = swin_t(weights=weights)

#     num_classes = len(class_names)
#     model.head = nn.Linear(model.head.in_features, num_classes)

#     model.load_state_dict(torch.load("swin_lung_model.pth", map_location=device))
#     model = model.to(device)
#     model.eval()
#     return model, weights, device

# model, weights, device = load_model()

uploaded_file = st.file_uploader(" Upload Chest Scan Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Scan", use_container_width=True)

    # Preprocess image and predict
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])


    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0].cpu().numpy()

    pred_index = torch.argmax(torch.tensor(probabilities)).item()
    prediction = class_names[pred_index]
    confidence = probabilities[pred_index] * 100

    # Display prediction results
    st.markdown(f"## Prediction: **{prediction}**")
    st.metric("Confidence", f"{confidence:.2f}%")


    # Show class probabilities chart
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(class_names, probabilities * 100, color="teal")
    ax.set_xlabel("Confidence (%)")
    ax.set_title("Prediction Confidence per Class")
    plt.tight_layout()
    st.pyplot(fig)


    if prediction == "Normal":
        st.success("✅ The scan appears **Normal**.")
    else:
        st.warning(f"⚠️ Detected possible signs of **{prediction}**. Please consult a medical professional.")
        
        
st.markdown("---")
st.caption("Developed by Ojo Caleb — Data Scientist & ML Engineer")

