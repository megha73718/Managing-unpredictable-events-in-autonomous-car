Steps to follow for the project Execution

Installation Accepts image or video input from the user.
Uses YOLOv8 to detect and label objects like cars, people, etc.

Uses CLIP to classify the overall driving scene context using textual prompts.

Displays the image/video back to the user with:

Object bounding boxes

Scene description

Warnings for dangerous situations like accidents or hazards

This comment tells you what packages are required:

ultralytics: For YOLOv8 object detection

transformers: For using CLIP model from Hugging Face

opencv-python: For video/image processing

pillow: For image conversion

streamlit: For building the web UI

Import Libraries
import streamlit as st from ultralytics import YOLO from transformers import CLIPProcessor, CLIPModel from PIL import Image import torch import cv2 import numpy as np import tempfile import os These are all necessary imports:

YOLO: Object detection model from Ultralytics CLIPProcessor and CLIPModel: For scene classification cv2, PIL, torch, etc., for image and video processing streamlit for the web app interface

Run modules
Runs YOLOv8 detection on the input frame.

Draws bounding boxes and labels around detected objects.

Classifies the scene with CLIP (calls previous function).

Checks for risks: if CLIP says the scene is dangerous and has high confidence, it prepares a warning message.

Returns the annotated frame, detected classes, CLIP classification, and warning.

Summary

Accepts image or video input from the user.

Uses YOLOv8 to detect and label objects like cars, people, etc.

Uses CLIP to classify the overall driving scene context using textual prompts.

Displays the image/video back to the user with:

Object bounding boxes

Scene description

Warnings for dangerous situations like accidents or hazards
