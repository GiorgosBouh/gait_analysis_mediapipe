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


def detect_heel_strikes(y_positions):
    peaks, _ = find_peaks(-np.array(y_positions), distance=10)
    return peaks


def run_live_gait_analysis():
    st.title("🎥 Live Gait Camera Preview")
    st.markdown("Adjust your position and framing. Then press **Start Recording** when ready.")

    if "recording" not in st.session_state:
        st.session_state.recording = False
    if "recorded_data" not in st.session_state:
        st.session_state.recorded_data = {
            "left_heel_x": [], "left_heel_y": [],
            "right_heel_x": [], "right_heel_y": [],
            "joints": {j: [] for j in [
                'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_HIP', 'RIGHT_HIP', 'LEFT_ANKLE', 'RIGHT_ANKLE']},
            "angles": {j: [] for j in [
                'left_knee', 'right_knee', 'left_hip', 'right_hip', 'left_ankle', 'right_ankle']},
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
                    "left_heel_x": [], "left_heel_y": [],
                    "right_heel_x": [], "right_heel_y": [],
                    "joints": {j: [] for j in st.session_state.recorded_data["joints"]},
                    "angles": {j: [] for j in st.session_state.recorded_data["angles"]},
                    "frame_idx": 0,
                    "csv_path": None,
                    "video_path": None
                }
    with col2:
        if st.session_state.recording:
            if st.button("⏹️ Stop Recording"):
                st.session_state.recording = False

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("❌ Cannot open webcam. Make sure it is connected and accessible.")
        return

    width, height = int(cap.get(3)), int(cap.get(4))
    frame_display = st.empty()
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mp_drawing = mp.solutions.drawing_utils

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
                    lm = result.pose_landmarks.landmark
                    row = [st.session_state.recorded_data["frame_idx"]] + [v for landmark in lm for v in [landmark.x, landmark.y, landmark.z, landmark.visibility]]
                    csv_writer.writerow(row)

                    st.session_state.recorded_data["left_heel_x"].append(lm[mp_pose.PoseLandmark.LEFT_HEEL].x)
                    st.session_state.recorded_data["left_heel_y"].append(lm[mp_pose.PoseLandmark.LEFT_HEEL].y)
                    st.session_state.recorded_data["right_heel_x"].append(lm[mp_pose.PoseLandmark.RIGHT_HEEL].x)
                    st.session_state.recorded_data["right_heel_y"].append(lm[mp_pose.PoseLandmark.RIGHT_HEEL].y)

                    for joint in st.session_state.recorded_data["joints"]:
                        st.session_state.recorded_data["joints"][joint].append([lm[mp_pose.PoseLandmark[joint]].x, lm[mp_pose.PoseLandmark[joint]].y, lm[mp_pose.PoseLandmark[joint]].z])

                    st.session_state.recorded_data["angles"]["left_knee"].append(
                        calculate_angle(lm[mp_pose.PoseLandmark.LEFT_HIP], lm[mp_pose.PoseLandmark.LEFT_KNEE], lm[mp_pose.PoseLandmark.LEFT_ANKLE]))
                    st.session_state.recorded_data["angles"]["right_knee"].append(
                        calculate_angle(lm[mp_pose.PoseLandmark.RIGHT_HIP], lm[mp_pose.PoseLandmark.RIGHT_KNEE], lm[mp_pose.PoseLandmark.RIGHT_ANKLE]))
                    st.session_state.recorded_data["angles"]["left_hip"].append(
                        calculate_angle(lm[mp_pose.PoseLandmark.LEFT_SHOULDER], lm[mp_pose.PoseLandmark.LEFT_HIP], lm[mp_pose.PoseLandmark.LEFT_KNEE]))
                    st.session_state.recorded_data["angles"]["right_hip"].append(
                        calculate_angle(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER], lm[mp_pose.PoseLandmark.RIGHT_HIP], lm[mp_pose.PoseLandmark.RIGHT_KNEE]))
                    st.session_state.recorded_data["angles"]["left_ankle"].append(
                        calculate_angle(lm[mp_pose.PoseLandmark.LEFT_KNEE], lm[mp_pose.PoseLandmark.LEFT_ANKLE], lm[mp_pose.PoseLandmark.LEFT_HEEL]))
                    st.session_state.recorded_data["angles"]["right_ankle"].append(
                        calculate_angle(lm[mp_pose.PoseLandmark.RIGHT_KNEE], lm[mp_pose.PoseLandmark.RIGHT_ANKLE], lm[mp_pose.PoseLandmark.RIGHT_HEEL]))

                    st.session_state.recorded_data["frame_idx"] += 1

        cap.release()
        out.release()
        csv_file.close()
        st.session_state.recorded_data.update({"csv_path": csv_file_path, "video_path": video_file})
        st.success("✅ Recording complete. Gait features will now be shown below.")
    else:
        ret, frame = cap.read()
        if ret:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)
            if result.pose_landmarks:
                mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            frame_display.image(frame, channels="BGR", use_container_width=True)
        cap.release()

    if not st.session_state.recording and st.session_state.recorded_data["frame_idx"] > 0:
        rd = st.session_state.recorded_data
        fps = 30
        duration = rd["frame_idx"] / fps
        left_peaks = detect_heel_strikes(rd["left_heel_y"])
        right_peaks = detect_heel_strikes(rd["right_heel_y"])

        def mean_stride(peaks, x):
            strides = [np.abs(x[peaks[i + 1]] - x[peaks[i]]) for i in range(len(peaks) - 1)]
            return np.mean(strides) if strides else 0

        left_stride = mean_stride(left_peaks, rd["left_heel_x"])
        right_stride = mean_stride(right_peaks, rd["right_heel_x"])
        left_speed = left_stride / (duration / len(left_peaks)) if len(left_peaks) > 0 else 0
        right_speed = right_stride / (duration / len(right_peaks)) if len(right_peaks) > 0 else 0

        st.subheader("👟 Stride Characteristics per Leg")
        df = pd.DataFrame({
            "Leg": ["Left", "Right"],
            "Stride Length (m)": [f"{left_stride:.2f}", f"{right_stride:.2f}"],
            "Stride Speed (m/s)": [f"{left_speed:.2f}", f"{right_speed:.2f}"]
        })
        st.dataframe(df, use_container_width=True)

        st.subheader("🦵 Joint ROM (Degrees)")
        for joint, angles in rd["angles"].items():
            if angles:
                rom = max(angles) - min(angles)
                st.markdown(f"- **{joint.replace('_', ' ').title()} ROM:** `{rom:.2f}°`")

        st.subheader("📈 Heel Strike Detection")
        fig, ax = plt.subplots()
        ax.plot(rd["left_heel_y"], label='Left Heel Y')
        ax.plot(left_peaks, np.array(rd["left_heel_y"])[left_peaks], "rx", label='Left HS')
        ax.plot(rd["right_heel_y"], label='Right Heel Y')
        ax.plot(right_peaks, np.array(rd["right_heel_y"])[right_peaks], "bx", label='Right HS')
        ax.set_title("Heel Strike Events")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Y Position")
        ax.legend()
        st.pyplot(fig)

        st.info(f"📄 CSV: `{rd['csv_path']}`\n🎥 Video: `{rd['video_path']}`")