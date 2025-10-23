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

MODEL_PATH = "swin_lung_model.pth"

# CORRECTED GOOGLE DRIVE URL FORMATS:
# Option 1: Direct download URL (use this one)
DRIVE_URL = "https://drive.google.com/uc?id=1SWOehqN5jmJW0t90b9llUxngrhhlCfNT"

# Option 2: Alternative format
# DRIVE_URL = "https://drive.google.com/uc?export=download&id=1SWOehqN5jmJW0t90b9llUxngrhhlCfNT"

@st.cache_resource
def load_model():
    """Load the model, download if not present"""
    # Download model if it doesn't exist
    if not os.path.exists(MODEL_PATH):
        with st.spinner("📥 Downloading model from Google Drive... (This may take a few minutes for 100MB+ file)"):
            try:
                # Method 1: Using gdown with correct URL
                gdown.download(DRIVE_URL, MODEL_PATH, quiet=False)
                
                # Method 2: If above fails, try with fuzzy match
                # gdown.download("https://drive.google.com/file/d/1SWOehqN5jmJW0t90b9llUxngrhhlCfNT/view?usp=drive_link", 
                #               MODEL_PATH, fuzzy=True, quiet=False)
                
                # Check if file was downloaded properly
                if os.path.exists(MODEL_PATH):
                    file_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)  # Size in MB
                    st.success(f"✅ Model downloaded successfully! Size: {file_size:.1f} MB")
                    
                    # Check if file size is reasonable
                    if file_size < 50:  # If less than 50MB, probably wrong file or incomplete download
                        st.warning(f"⚠️ Downloaded file seems small ({file_size:.1f} MB). Expected ~105MB.")
                        st.info("This might be the wrong file or the download was incomplete.")
                        return None, None
                else:
                    st.error("❌ Model file not found after download attempt")
                    return None, None
                    
            except Exception as e:
                st.error(f"❌ Error downloading model: {str(e)}")
                st.info("Trying alternative download method...")
                
                # Alternative method using requests
                try:
                    import requests
                    direct_url = "https://drive.google.com/uc?export=download&id=1SWOehqN5jmJW0t90b9llUxngrhhlCfNT"
                    response = requests.get(direct_url, stream=True)
                    
                    # Handle large file download
                    with open(MODEL_PATH, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                    
                    if os.path.exists(MODEL_PATH):
                        file_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)
                        st.success(f"✅ Model downloaded via alternative method! Size: {file_size:.1f} MB")
                    else:
                        st.error("❌ Alternative download also failed")
                        return None, None
                        
                except Exception as e2:
                    st.error(f"❌ All download methods failed: {str(e2)}")
                    return None, None
    
    # Load the model
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        st.info(f"Using device: {device}")
        
        # Check file size before loading
        if os.path.exists(MODEL_PATH):
            file_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)
            st.write(f"Model file size: {file_size:.1f} MB")
            
            # If file is still small, use demo mode
            if file_size < 50:
                st.warning("Using demo mode due to small model file size")
                return "demo", device
        
        # Create model architecture
        model = swin_t(weights=None)
        model.head = nn.Linear(model.head.in_features, len(class_names))
        
        # Load state dict
        state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        model.load_state_dict(state_dict)
        
        model = model.to(device)
        model.eval()
        
        st.success("✅ Model loaded successfully!")
        return model, device
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return "demo", device  # Fall back to demo mode

def demo_prediction(image):
    """Generate demo predictions when real model isn't available"""
    import numpy as np
    
    # Generate realistic mock probabilities
    img_array = np.array(image)
    
    # Simple heuristic based on image properties
    if len(img_array.shape) == 3:
        brightness = np.mean(img_array)
        contrast = np.std(img_array)
        
        if brightness > 150 and contrast < 50:
            base_probs = [0.7, 0.1, 0.1, 0.05, 0.05]  # Likely normal
        else:
            base_probs = [0.3, 0.2, 0.2, 0.2, 0.1]   # More varied
    else:
        base_probs = [0.25, 0.25, 0.25, 0.15, 0.1]
    
    # Add some randomness but keep it stable for the same file
    import hashlib
    seed = int(hashlib.md5(image.tobytes()).hexdigest()[:8], 16) % 10000
    np.random.seed(seed)
    noise = np.random.normal(0, 0.1, 5)
    probs = np.abs(np.array(base_probs) + noise)
    probs = probs / np.sum(probs)
    
    return probs

# Load model
model_info = load_model()
if model_info is None:
    model, device = None, None
else:
    model, device = model_info

# File uploader
uploaded_file = st.file_uploader("📁 Upload Chest Scan Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(image, caption="Uploaded Scan", use_container_width=True)
    
    with col2:
        if model == "demo" or model is None:
            st.warning("🔧 Running in Demo Mode")
            st.info("Real model file not available. Showing demo predictions.")
            
            # Demo predictions
            probabilities = demo_prediction(image)
            pred_index = np.argmax(probabilities)
            prediction = class_names[pred_index]
            confidence = probabilities[pred_index] * 100
            
        else:
            # Real model predictions
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            ])
            
            try:
                input_tensor = transform(image).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    outputs = model(input_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)[0].cpu().numpy()
                
                pred_index = np.argmax(probabilities)
                prediction = class_names[pred_index]
                confidence = probabilities[pred_index] * 100
                
            except Exception as e:
                st.error(f"❌ Prediction error: {str(e)}")
                st.info("Falling back to demo mode")
                probabilities = demo_prediction(image)
                pred_index = np.argmax(probabilities)
                prediction = class_names[pred_index]
                confidence = probabilities[pred_index] * 100
        
        # Display results
        st.markdown(f"## 🎯 Prediction: **{prediction}**")
        st.metric("Confidence", f"{confidence:.2f}%")
        
        # Show confidence for all classes
        st.subheader("📊 Confidence Scores")
        for class_name, prob in zip(class_names, probabilities):
            progress_value = min(int(prob * 100), 100)
            st.write(f"**{class_name}**: {progress_value}%")
            st.progress(progress_value)
        
        # Visualization
        fig, ax = plt.subplots(figsize=(8, 4))
        y_pos = range(len(class_names))
        colors = ['lightgreen' if i == pred_index else 'lightblue' for i in range(len(class_names))]
        ax.barh(y_pos, probabilities * 100, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(class_names)
        ax.set_xlabel('Confidence (%)')
        ax.set_title('Prediction Confidence per Class')
        plt.tight_layout()
        st.pyplot(fig)
        
        # Medical advice
        if prediction == "Normal":
            st.success("✅ The scan appears **Normal**. No signs of disease detected.")
        else:
            st.warning(f"⚠️ **Important**: Possible signs of **{prediction}** detected. Consult a medical professional.")

# Debug information
with st.expander("🔧 System Info"):
    if os.path.exists(MODEL_PATH):
        file_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)
        st.write(f"Model file size: {file_size:.2f} MB")
    st.write(f"Model status: {'Real model' if model not in ['demo', None] else 'Demo mode'}")
    st.write(f"Device: {device}")

st.markdown("---")
st.caption("Developed by Ojo Caleb — Data Scientist & ML Engineer")