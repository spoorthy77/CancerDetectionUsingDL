import gradio as gr
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the models
stage1_model = load_model("tumor_presence_mobilenet.h5")  # expects RGB
stage2_model = load_model("tumor_type_classifier.h5")     # expects grayscale

tumor_type_labels = ['glioma', 'meningioma', 'pituitary']

# Preprocess for Stage 1 (RGB)
def preprocess_stage1(image_path):
    img = cv2.imread(image_path)  # BGR
    if img is None:
        raise ValueError("Image not found.")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    return img.reshape(1, 224, 224, 3)

# Preprocess for Stage 2 (Grayscale)
def preprocess_stage2(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Image not found.")
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    return img.reshape(1, 224, 224, 1)

# Prediction logic
def predict_tumor(image_path):
    # Stage 1: Tumor detection
    img_rgb = preprocess_stage1(image_path)
    stage1_pred = stage1_model.predict(img_rgb)[0][0]
    has_tumor = stage1_pred > 0.5
    conf = round(float(stage1_pred), 2)

    result = f"🧠 Tumor Detected (confidence: {conf}) ✅" if has_tumor else f"🧼 No Tumor Detected (confidence: {1 - conf}) ✅"

    # Stage 2: Tumor type
    if has_tumor:
        img_gray = preprocess_stage2(image_path)
        type_pred = stage2_model.predict(img_gray)[0]
        tumor_index = np.argmax(type_pred)
        tumor_name = tumor_type_labels[tumor_index]
        tumor_conf = round(float(type_pred[tumor_index]), 2)
        result += f"\n🔍 Tumor Type: {tumor_name} ({tumor_conf})"

    return result

# Gradio UI
demo = gr.Interface(
    fn=predict_tumor,
    inputs=gr.Image(type="filepath", label="📤 Upload MRI Image"),
    outputs="text",
    title="🧠 Brain Tumor Detection & Classification",
    description="Upload an MRI to detect tumor presence and type using deep learning (Stage 1: MobileNetV2 + Stage 2: CNN)."
)

if __name__ == "__main__":
    demo.launch()
