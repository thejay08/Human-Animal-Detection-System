import streamlit as st
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import tempfile
import os
import matplotlib.pyplot as plt

# Load CLIP model and processor
@st.cache_resource
def load_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

# Define candidate labels
CANDIDATE_LABELS = [
    "a photo of a human",
    "a photo of a dog",
    "a photo of a cat",
    "a photo of a cow",
    "a photo of a goat",
    "a photo of a lion",
    "a photo of a tiger",
    "a photo of a bear",
    "a photo of a horse",
    "a photo of a deer",
    "a photo of a bird",
    "a photo of a snake",
    "a photo of a background with no animals or humans"
]

# Detect entities
def detect_entities(image, model, processor, candidate_labels=CANDIDATE_LABELS):
    inputs = processor(text=candidate_labels, images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)

    results = [
        {"label": label.replace("a photo of a ", ""), "score": probs[0][i].item()}
        for i, label in enumerate(candidate_labels)
    ]
    return sorted(results, key=lambda x: x["score"], reverse=True)

# Process a single image
def process_image(image_data, model, processor, threshold=0.5):
    image = Image.open(image_data).convert("RGB")
    results = detect_entities(image, model, processor)

    top_result = results[0]
    alert_msg = None
    if "human" in top_result["label"] and top_result["score"] > threshold:
        alert_msg = "⚠️ ALERT: Human detected!"
    elif top_result["label"] != "background with no animals or humans" and top_result["score"] > threshold:
        alert_msg = f"🦁 ALERT: Animal detected! ({top_result['label']})"

    return image, results[:3], alert_msg

# Process a video
def process_video(video_path, model, processor, frame_interval=30, max_frames=5):
    cap = cv2.VideoCapture(video_path)
    results_summary = []

    frame_count = 0
    processed = 0
    while cap.isOpened() and processed < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            results = detect_entities(image, model, processor)
            results_summary.append((frame_count, results[:3]))
            processed += 1

        frame_count += 1

    cap.release()
    return results_summary

# Streamlit App
st.set_page_config(page_title="👁️ AI Entity Detector", layout="wide")
st.title("🔍 AI-Powered Human & Animal Detector")
st.caption("Powered by OpenAI CLIP model | Built with ❤️ by Jay")

# Load model
model, processor = load_model()

tab1, tab2 = st.tabs(["🖼️ Image Detection", "🎥 Video Detection"])

with tab1:
    st.header("Upload an Image")
    uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        image, top_results, alert = process_image(uploaded_image, model, processor)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        if alert:
            st.warning(alert)
        st.subheader("Top 3 Predictions")
        for i, res in enumerate(top_results):
            st.markdown(f"**{i+1}. {res['label'].capitalize()}** — Confidence: `{res['score']:.4f}`")

with tab2:
    st.header("Upload a Video")
    uploaded_video = st.file_uploader("Choose a video", type=["mp4", "mov", "avi"])
    if uploaded_video:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_video.read())
            tmp_path = tmp_file.name
        st.video(uploaded_video)
        st.info("Processing video... (extracting frames)")
        summary = process_video(tmp_path, model, processor)

        st.subheader("Video Frame Classification Results")
        for frame_num, res in summary:
            st.markdown(f"**Frame {frame_num}**")
            for i, item in enumerate(res):
                st.markdown(f"- {i+1}. {item['label'].capitalize()} — `{item['score']:.4f}`")
            st.markdown("---")
        os.remove(tmp_path)
