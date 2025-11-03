import os
import cv2
import numpy as np
import tempfile
from tensorflow.keras.models import load_model
from PIL import Image
import gradio as gr

# Load models
stage1_model = load_model("tumor_presence_mobilenet.h5")     # Trained on RGB images
stage2_model = load_model("tumor_type_classifier.h5")        # Trained on RGB images

# Tumor type class labels
tumor_types = {0: "meningioma", 1: "glioma", 2: "pituitary"}

# Unified Preprocessing Function (RGB, 3 channels)
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found")
    
    # Convert grayscale to RGB if needed
    if len(img.shape) == 2 or img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    return img.reshape(1, 224, 224, 3)

# Main prediction function
def predict(image):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmpfile:
        temp_path = tmpfile.name
        image.save(temp_path)

    try:
        # Stage 1: Tumor Presence Detection
        img_stage1 = preprocess_image(temp_path)
        tumor_prob = stage1_model.predict(img_stage1)[0][0]

        if tumor_prob < 0.5:
            return (
                "🧠 No Tumor Detected ❌",
                f"Confidence: {1 - tumor_prob:.2f}",
                "-",
                image
            )

        # Stage 2: Tumor Type Classification
        img_stage2 = preprocess_image(temp_path)
        probs = stage2_model.predict(img_stage2)[0]
        pred_idx = np.argmax(probs)
        tumor_type = tumor_types[pred_idx]
        confidence = probs[pred_idx]

        return (
            "🧠 Tumor Detected ✅",
            f"Confidence: {tumor_prob:.2f}",
            f"Tumor Type: {tumor_type} ({confidence:.2f})",
            image
        )

    except Exception as e:
        print("Error:", str(e))  # Log error in console
        return (
            "🚨 Prediction Error",
            "—",
            "—",
            image
        )
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

# Gradio UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 🧠 Brain Tumor Detection & Classification")
    gr.Markdown("Upload an MRI to detect tumor presence and classify its type.")

    with gr.Row():
        image_input = gr.Image(type="pil", label="📤 Upload MRI Image")
        image_output = gr.Image(type="pil", label="🖼️ Preview")

    with gr.Row():
        with gr.Column():
            result1 = gr.Label(label="Stage 1: Tumor Detection")
            result2 = gr.Label(label="Detection Confidence")
        with gr.Column():
            result3 = gr.Label(label="Stage 2: Tumor Type")

    predict_button = gr.Button("🔍 Run Prediction")

    predict_button.click(
        fn=predict,
        inputs=[image_input],
        outputs=[result1, result2, result3, image_output]
    )

demo.launch()