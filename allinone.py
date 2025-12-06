# ✅ INSTALL DEPENDENCIES (Run in terminal if needed)
# pip install ultralytics transformers opencv-python pillow streamlit

import streamlit as st
from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import cv2
import numpy as np
import tempfile
import os

# Load Models
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
yolo_model = YOLO("yolov8n.pt")  # Lightweight YOLOv8 model for fast inference

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model.to(device)

# Zero-shot prompts for scene classification
prompts = [
    "a safe driving scene",
    "a pedestrian is crossing suddenly",
    "a vehicle is about to crash",
    "two vehicles have collided",
    "a child might run into the road",
    "a cyclist is dangerously close",
    "a fallen object is on the road",
    "a fire or smoke ahead",
    "a construction site on the road",
    "an animal crossing the road",
    "a dangerous situation ahead"
]

# Classify scene with CLIP
@st.cache_resource
def classify_scene_with_clip(frame):
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = clip_processor(text=prompts, images=pil_image, return_tensors="pt", padding=True).to(device)
    outputs = clip_model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1).squeeze().detach().cpu().numpy()
    ranked = sorted(zip(prompts, probs), key=lambda x: x[1], reverse=True)
    return ranked

# Analyze frame: detect objects with YOLO, draw boxes, classify scene with CLIP
@st.cache_data(show_spinner=False)
def analyze_frame(frame):
    results = yolo_model(frame, verbose=False)[0]  # YOLOv8 detection

    classes = results.names
    detected = [classes[int(cls)] for cls in results.boxes.cls.cpu().numpy()] if results.boxes is not None else []

    # Copy frame to draw annotations
    annotated_frame = frame.copy()

    # Draw bounding boxes and labels
    if results.boxes is not None:
        for box in results.boxes:
            xyxy = box.xyxy.cpu().numpy().astype(int)[0]  # [x1, y1, x2, y2]
            cls = int(box.cls.cpu().numpy())
            conf = box.conf.cpu().numpy()[0]
            label = f"{classes[cls]} {conf:.2f}"

            # Draw rectangle box
            cv2.rectangle(annotated_frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)

            # Draw label background
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(annotated_frame, (xyxy[0], xyxy[1] - 20), (xyxy[0] + w, xyxy[1]), (0, 255, 0), -1)

            # Put label text
            cv2.putText(annotated_frame, label, (xyxy[0], xyxy[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    # Classify scene with CLIP
    clip_results = classify_scene_with_clip(frame)
    top_clip = clip_results[0][0]
    top_prob = clip_results[0][1]

    warning = None
    if top_prob > 0.5 and top_clip != "a safe driving scene":
        if top_clip == "two vehicles have collided":
            warning = "🚨 Collision Detected: Call Emergency Services!"
        else:
            warning = f"⚠️ Warning: {top_clip}"

    return annotated_frame, detected, clip_results, warning, results

# Streamlit UI
st.set_page_config(page_title="Scene Risk Analyzer", layout="centered")
st.title("🚗 Driving Scene Risk Analyzer")
st.markdown("Upload an image or video to analyze potential driving risks using YOLO + CLIP.")

input_type = st.radio("Select Input Type", ["Image", "Video"])
file = st.file_uploader("Upload file", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])

if file:
    if input_type == "Image":
        image = Image.open(file).convert("RGB")
        frame = np.array(image)[:, :, ::-1]  # Convert RGB to BGR for OpenCV
        analyzed_frame, objects, clip_scores, warning, _ = analyze_frame(frame)

        analyzed_frame_rgb = cv2.cvtColor(analyzed_frame, cv2.COLOR_BGR2RGB)  # Convert back to RGB for display
        st.image(analyzed_frame_rgb, caption=f"Top Scene: {clip_scores[0][0]} ({clip_scores[0][1]:.2f})", use_container_width=True)

        if warning:
            st.error(warning)
        else:
            st.success("✅ Safe driving scene")

    elif input_type == "Video":
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(file.read())
        cap = cv2.VideoCapture(tfile.name)

        frame_placeholder = st.empty()
        info_placeholder = st.empty()
        warning_placeholder = st.empty()
        warning_flag = False

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (640, 480))
            analyzed_frame, _, clip_scores, warning, _ = analyze_frame(frame)
            frame_rgb = cv2.cvtColor(analyzed_frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            info_placeholder.info(f"Top Scene: {clip_scores[0][0]} ({clip_scores[0][1]:.2f})")
            if warning and not warning_flag:
                warning_placeholder.warning(warning)
                warning_flag = True

        cap.release()
        os.unlink(tfile.name)
