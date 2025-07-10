# --- gait_live.py ---
import cv2
import mediapipe as mp
import os
import csv
from datetime import datetime
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from math import acos, degrees


def calculate_angle(a, b, c):
    a = np.array([a.x, a.y, a.z])
    b = np.array([b.x, b.y, b.z])
    c = np.array([c.x, c.y, c.z])
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = degrees(acos(np.clip(cosine_angle, -1.0, 1.0)))
    return angle


def run_live_gait_analysis():
    st.title("🎥 Live Gait Camera Preview")
    st.markdown("Adjust your position and framing. Then press **Start Recording** when ready.")

    if "recording" not in st.session_state:
        st.session_state.recording = False
    if "recorded_data" not in st.session_state:
        st.session_state.recorded_data = {
            "left_xs": [], "left_ys": [],
            "right_xs": [], "right_ys": [],
            "joints": {
                'LEFT_KNEE': [], 'RIGHT_KNEE': [],
                'LEFT_HIP': [], 'RIGHT_HIP': [],
                'LEFT_ANKLE': [], 'RIGHT_ANKLE': []
            },
            "angles": {
                'left_knee': [], 'right_knee': [],
                'left_hip": [], 'right_hip': [],
                'left_ankle': [], 'right_ankle': []
            },
            "frame_idx": 0,
            "csv_path": None,
            "video_path": None
        }

    zoom = st.slider("Zoom level (simulated crop)", 1.0, 2.0, 1.0, step=0.1)

    col1, col2 = st.columns(2)
    with col1:
        if not st.session_state.recording:
            if st.button("▶️ Start Recording"):
                st.session_state.recording = True
                st.session_state.recorded_data = {
                    "left_xs": [], "left_ys": [],
                    "right_xs": [], "right_ys": [],
                    "joints": {k: [] for k in st.session_state.recorded_data["joints"]},
                    "angles": {k: [] for k in st.session_state.recorded_data["angles"]},
                    "frame_idx": 0,
                    "csv_path": None,
                    "video_path": None
                }
    with col2:
        if st.session_state.recording:
            if st.button("⏹️ Stop Recording"):
                st.session_state.recording = False

    # MediaPipe setup
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mp_drawing = mp.solutions.drawing_utils

    # Always open and preview camera frame
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("❌ Cannot open webcam. Make sure it is connected and accessible.")
        return

    width, height = int(cap.get(3)), int(cap.get(4))
    frame_display = st.empty()

    ret, frame = cap.read()
    if ret:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)
        if result.pose_landmarks:
            mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        frame_display.image(frame, channels="BGR", use_container_width=True)

    # If recording starts
    if st.session_state.recording:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("outputs", exist_ok=True)
        video_file = f"outputs/gait_live_{timestamp}.mp4"
        csv_file_path = f"outputs/gait_live_{timestamp}.csv"

        out = cv2.VideoWriter(video_file, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
        csv_file = open(csv_file_path, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        landmark_names = [l.name for l in mp_pose.PoseLandmark]
        header = ['frame'] + [f"{n}_{a}" for n in landmark_names for a in ['x', 'y', 'z', 'visibility']]
        csv_writer.writerow(header)

        with st.spinner("Recording... Press Stop to finish."):
            while st.session_state.recording:
                ret, frame = cap.read()
                if not ret:
                    break

                if zoom > 1.0:
                    center_x, center_y = width // 2, height // 2
                    new_w, new_h = int(width / zoom), int(height / zoom)
                    left, top = center_x - new_w // 2, center_y - new_h // 2
                    frame = frame[top:top + new_h, left:left + new_w]
                    frame = cv2.resize(frame, (width, height))

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = pose.process(rgb)
                if result.pose_landmarks:
                    mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                frame_display.image(frame, channels="BGR", use_container_width=True)
                out.write(frame)

                if result.pose_landmarks:
                    row = [st.session_state.recorded_data["frame_idx"]]
                    for lm in result.pose_landmarks.landmark:
                        row.extend([lm.x, lm.y, lm.z, lm.visibility])
                    csv_writer.writerow(row)

                    lm = result.pose_landmarks.landmark
                    st.session_state.recorded_data["left_xs"].append(lm[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x)
                    st.session_state.recorded_data["left_ys"].append(lm[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y)
                    st.session_state.recorded_data["right_xs"].append(lm[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].x)
                    st.session_state.recorded_data["right_ys"].append(lm[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y)

                    for joint in st.session_state.recorded_data["joints"]:
                        st.session_state.recorded_data["joints"][joint].append([
                            lm[mp_pose.PoseLandmark[joint]].x,
                            lm[mp_pose.PoseLandmark[joint]].y,
                            lm[mp_pose.PoseLandmark[joint]].z
                        ])

                    st.session_state.recorded_data["angles"]["left_knee"].append(
                        calculate_angle(lm[mp_pose.PoseLandmark.LEFT_HIP], lm[mp_pose.PoseLandmark.LEFT_KNEE], lm[mp_pose.PoseLandmark.LEFT_ANKLE])
                    )
                    st.session_state.recorded_data["angles"]["right_knee"].append(
                        calculate_angle(lm[mp_pose.PoseLandmark.RIGHT_HIP], lm[mp_pose.PoseLandmark.RIGHT_KNEE], lm[mp_pose.PoseLandmark.RIGHT_ANKLE])
                    )

                    st.session_state.recorded_data["frame_idx"] += 1

        cap.release()
        out.release()
        csv_file.close()

        st.session_state.recorded_data["csv_path"] = csv_file_path
        st.session_state.recorded_data["video_path"] = video_file
        st.success("✅ Recording complete. Gait features will now be shown below.")
    else:
        cap.release()