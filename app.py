# import streamlit as st
# import torch
# import torch.nn as nn
# from torchvision.models import swin_t, Swin_T_Weights
# from torchvision import transforms
# from PIL import Image
# import matplotlib.pyplot as plt
# import os
# import requests


# st.title("**Lung Disease Detection using Swin Transformer**")
# st.write("Upload a chest CT scan or X-ray to predict the type of lung disease using a fine-tuned **Swin Transformer** model.")

# class_names = [
#     "Normal",
#     "Large Cell Carcinoma",
#     "Squamous Cell Carcinoma",
#     "Adenocarcinoma",
#     "Benign"
# ]

# MODEL_PATH = "model.pth"
# DRIVE_URL = "https://drive.google.com/file/d/1SWOehqN5jmJW0t90b9llUxngrhhlCfNT/view?usp=sharing"


# from torchvision.models import swin_t, Swin_T_Weights
# import torch
# import os
# import requests
# import streamlit as st

# class_names = ["Normal", "Large Cell Carcinoma", "Squamous Cell Carcinoma", "Adenocarcinoma", "Benign"]
# MODEL_PATH = "swin_lung_model.pth"
# DRIVE_URL = "https://drive.google.com/uc?id=YOUR_FILE_ID"  # 👈 replace with actual file ID


# @st.cache_resource
# def load_model():
#     if not os.path.exists(MODEL_PATH):
#         with st.spinner("Downloading model from Google Drive... (only once)"):
#             r = requests.get(DRIVE_URL)
#             open(MODEL_PATH, "wb").write(r.content)
#             st.success("Model downloaded successfully!")

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     weights = Swin_T_Weights.DEFAULT
#     model = swin_t(weights=None)
#     model.head = torch.nn.Linear(model.head.in_features, len(class_names))

#     state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=False)
#     model.load_state_dict(state_dict)

#     model = model.to(device)
#     model.eval()
#     return model


# model = load_model()

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # @st.cache_resource
# # def load_model():
# #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# #     weights = Swin_T_Weights.DEFAULT
# #     model = swin_t(weights=weights)

# #     num_classes = len(class_names)
# #     model.head = nn.Linear(model.head.in_features, num_classes)

# #     model.load_state_dict(torch.load("swin_lung_model.pth", map_location=device))
# #     model = model.to(device)
# #     model.eval()
# #     return model, weights, device

# # model, weights, device = load_model()

# uploaded_file = st.file_uploader(" Upload Chest Scan Image", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     image = Image.open(uploaded_file).convert("RGB")
#     st.image(image, caption="Uploaded Scan", use_container_width=True)

#     # Preprocess image and predict
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                             std=[0.229, 0.224, 0.225])
#     ])


#     input_tensor = transform(image).unsqueeze(0).to(device)

#     with torch.no_grad():
#         outputs = model(input_tensor)
#         probabilities = torch.nn.functional.softmax(outputs, dim=1)[0].cpu().numpy()

#     pred_index = torch.argmax(torch.tensor(probabilities)).item()
#     prediction = class_names[pred_index]
#     confidence = probabilities[pred_index] * 100

#     # Display prediction results
#     st.markdown(f"## Prediction: **{prediction}**")
#     st.metric("Confidence", f"{confidence:.2f}%")


#     # Show class probabilities chart
#     fig, ax = plt.subplots(figsize=(7, 4))
#     ax.barh(class_names, probabilities * 100, color="teal")
#     ax.set_xlabel("Confidence (%)")
#     ax.set_title("Prediction Confidence per Class")
#     plt.tight_layout()
#     st.pyplot(fig)


#     if prediction == "Normal":
#         st.success("✅ The scan appears **Normal**.")
#     else:
#         st.warning(f"⚠️ Detected possible signs of **{prediction}**. Please consult a medical professional.")
        
        
# st.markdown("---")
# st.caption("Developed by Ojo Caleb — Data Scientist & ML Engineer")


# =================================================================================

import streamlit as st
import torch
import torch.nn as nn
from torchvision.models import swin_t
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
import requests
import gdown  

# Set page config first
st.set_page_config(
    page_title="Lung Disease Detection",
    page_icon="🫁",
    layout="wide"
)

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

@st.cache_resource
def load_model():
    """Load the model, download if not present"""
    # Download model if it doesn't exist
    if not os.path.exists(MODEL_PATH):
        with st.spinner(" Downloading model from Google Drive... (This may take a few minutes)"):
            try:
                gdown.download(DRIVE_URL, MODEL_PATH, quiet=False)
                
                # Alternative Method 2: Using requests (if gdown fails)
                # session = requests.Session()
                # response = session.get(DRIVE_URL, stream=True)
                # if response.status_code == 200:
                #     with open(MODEL_PATH, 'wb') as f:
                #         for chunk in response.iter_content(chunk_size=32768):
                #             if chunk:
                #                 f.write(chunk)
                # else:
                #     st.error(f"Failed to download model. Status code: {response.status_code}")
                #     return None
                    
                st.success("Model downloaded successfully!")
            except Exception as e:
                st.error(f"Error downloading model: {str(e)}")
                st.info("Please check: 1) File ID is correct 2) File is publicly accessible")
                return None
    
    # Load the model
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        st.info(f"Using device: {device}")
        
        # Create model architecture
        model = swin_t(weights=None)
        model.head = nn.Linear(model.head.in_features, len(class_names))
        
        # Load state dict
        state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        model.load_state_dict(state_dict)
        
        model = model.to(device)
        model.eval()
        
        st.success("Model loaded successfully!")
        return model, device
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

# Load model
model, device = load_model()

# File uploader
uploaded_file = st.file_uploader("Upload Chest Scan Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    # Display image
    image = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(image, caption="Uploaded Scan", use_container_width=True)
    
    with col2:
        # Preprocess image
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
        
        try:
            input_tensor = transform(image).unsqueeze(0).to(device)
            
            # Make prediction
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0].cpu().numpy()
            
            pred_index = torch.argmax(torch.tensor(probabilities)).item()
            prediction = class_names[pred_index]
            confidence = probabilities[pred_index] * 100
            
            # Display results
            st.markdown(f"## Prediction: **{prediction}**")
            st.metric("Confidence", f"{confidence:.2f}%")
            
            # Show confidence for all classes
            st.subheader("Confidence Scores")
            for i, (class_name, prob) in enumerate(zip(class_names, probabilities)):
                progress = int(prob * 100)
                st.write(f"{class_name}: {progress}%")
                st.progress(progress)
            
            # Medical advice
            if prediction == "Normal":
                st.success("✅ The scan appears **Normal**. No signs of disease detected.")
            else:
                st.warning(f"⚠️ **Important**: Detected possible signs of **{prediction}**. This is an AI prediction and should be verified by a qualified medical professional.")
                
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")

elif uploaded_file is not None and model is None:
    st.error("Model not available. Please check the model download.")

# Information section
with st.expander("ℹ️ About this App"):
    st.markdown("""
    **How it works:**
    - Uses a Swin Transformer model fine-tuned on lung disease datasets
    - Analyzes chest CT scans and X-rays
    - Provides confidence scores for 5 different conditions
    
    **Supported Conditions:**
    - Normal
    - Large Cell Carcinoma
    - Squamous Cell Carcinoma  
    - Adenocarcinoma
    - Benign tumors
    
    **Disclaimer:** This tool is for educational purposes only. Always consult healthcare professionals for medical diagnoses.
    """)

st.markdown("---")
st.caption("Developed by Ojo Caleb — Data Scientist & ML Engineer")