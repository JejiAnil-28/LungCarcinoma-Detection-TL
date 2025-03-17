import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

# Load the trained model with caching
@st.cache_resource
def load_model():
    MODEL_PATH = "efficientnetb1_lung_cancer_model.h5"
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# Function to preprocess image
def preprocess_image(image):
    image = image.resize((224, 224)).convert("RGB")  # Resize & Convert to RGB
    image = np.array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Enhanced function to validate if image is likely a lung biopsy image
def is_likely_lung_biopsy(image, confidence_threshold=0.7):
    img_array = np.array(image)
    is_color = len(img_array.shape) == 3 and img_array.shape[2] == 3
    if is_color:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    color_std = np.std(img_array.reshape(-1, 3), axis=0).mean()
    if color_std < 30:
        return False, 0, "Color distribution suggests this is a diagram or chart"
    
    edges = cv2.Canny(gray, 50, 150)
    edge_percentage = np.count_nonzero(edges) / edges.size
    if edge_percentage > 0.1:
        return False, 0, "High edge density suggests this is a diagram or chart"
    
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    white_percentage = np.count_nonzero(thresh) / thresh.size
    if white_percentage > 0.7:
        return False, 0, "Large uniform areas suggest this is a diagram or chart"
    
    texture_measure = np.std(gray)
    if texture_measure < 20:
        return False, 0, "Texture analysis suggests this is not a biopsy image"
    
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    confidence = np.max(prediction) * 100
    
    if confidence > 99 and edge_percentage > 0.05:
        return False, confidence, "This appears to be a diagram with distinct edges"
    
    prediction_std = np.std(prediction[0])
    if prediction_std < 0.15:
        return False, confidence, "Model seems uncertain about classification"
    
    if confidence < confidence_threshold * 100:
        return False, confidence, "Low confidence prediction suggests this is not a lung biopsy image"
    
    return True, confidence, "Image appears to be a valid lung biopsy"

# Carcinoma details
details = {
    "Squamous cell carcinoma": {
        "Type": "Type 3 - High Risk",
        "Description": "A form of NSCLC that arises in squamous cells.",
        "Causes": "Smoking, carcinogens, chronic lung irritation.",
        "Treatment": "Surgery, radiation, chemotherapy, targeted therapy.",
        "Risk": "High risk - Aggressive growth."
    },
    "Adenocarcinoma": {
        "Type": "Type 2 - Moderate Risk",
        "Description": "A type of NSCLC in mucus-producing lung cells.",
        "Causes": "Smoking, pollution, genetic predisposition.",
        "Treatment": "Surgery, chemo, targeted therapy, immunotherapy.",
        "Risk": "Moderate risk - Common, treatable with early detection."
    },
    "Benign": {
        "Type": "Type 1 - Low Risk",
        "Description": "Non-cancerous lung tumors that do not spread.",
        "Causes": "Infections, inflammation, genetic factors.",
        "Treatment": "Usually monitored; surgery if symptomatic.",
        "Risk": "Low risk - Generally harmless."
    }
}

st.title("Lung Carcinoma Biopsy Image Classification")
st.write("Upload a lung biopsy image to classify lung carcinoma into Benign, Adenocarcinoma, or Squamous Cell Carcinoma.")

# Explanation of carcinoma types
with st.expander("Lung Carcinoma Types & Risk Levels"):
    st.markdown(
        """
        | **Type** | **Carcinoma Name** | **Description** | **Risk Level** |
        |------|----------------------------|-------------------------------------------------------------|------------|
        | **Type 1** | **Benign (Non-Cancerous)** | A non-cancerous tumor that does not spread to other body parts. Usually harmless but may require monitoring. | **Low Risk** (Generally harmless) |
        | **Type 2** | **Adenocarcinoma** | A type of **non-small cell lung cancer (NSCLC)** that starts in mucus-producing glandular cells. Common but treatable with early detection. | **Moderate Risk** (Common but treatable) |
        | **Type 3** | **Squamous Cell Carcinoma** | A form of **NSCLC** arising in squamous cells, strongly linked to smoking, with aggressive growth. | **High Risk** (Aggressive, potential spread) |
        """,
        unsafe_allow_html=True
    )

# Sidebar for extra info
st.sidebar.header("About This App")
st.sidebar.write(
    "This AI tool classifies lung biopsy images into:\n"
    "- **Benign (Non-cancerous)**\n"
    "- **Adenocarcinoma**\n"
    "- **Squamous Cell Carcinoma**\n\n"
    "Model: **EfficientNetB1** trained on biopsy images."
)


uploaded_files = st.file_uploader("Choose biopsy images...", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        with st.spinner("Processing..."):
            is_valid, confidence, reason = is_likely_lung_biopsy(image)
            if not is_valid:
                st.error("⚠️ This image does not appear to be a lung biopsy image.")
                st.warning(f"Reason: {reason}")
                continue
            
            processed_image = preprocess_image(image)
            prediction = model.predict(processed_image)
            class_labels = ["Squamous cell carcinoma", "Adenocarcinoma", "Benign"]
            predicted_class = class_labels[np.argmax(prediction)]
            confidence = np.max(prediction) * 100
            
            pred_type = details[predicted_class]["Type"]
            color = "#00b894" if predicted_class == "Benign" else "#d63031"
            st.markdown(f"""
                <div style="padding:10px; border-radius:10px; background-color:{color}; color:white; text-align:center;">
                    <h3>Prediction: {predicted_class} ({pred_type})</h3>
                    <h4>Confidence: {confidence:.2f}%</h4>
                </div>
                """, unsafe_allow_html=True)
            
            st.progress(int(confidence))
            
            st.subheader("What is it?")
            st.write(details[predicted_class]["Description"])
            
            st.subheader("Causes")
            st.write(details[predicted_class]["Causes"])
            
            st.subheader("Treatment")
            st.write(details[predicted_class]["Treatment"])
            
            st.subheader("Cancer Level")
            st.write(details[predicted_class]["Risk"])
