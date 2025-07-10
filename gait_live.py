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
import time

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

    zoom = st.slider("Zoom level (simulated crop)", 1.0, 2.0, 1.0, step=0.1)
    start = st.button("▶️ Start Recording")
    stop = st.button("⏹️ Stop Recording")

    if "recording" not in st.session_state:
        st.session_state.recording = False
    if "recorded_data" not in st.session_state:
        st.session_state.recorded_data = {}

    # Setup MediaPipe and webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("❌ Cannot open webcam.")
        return

    width, height = int(cap.get(3)), int(cap.get(4))
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mp_drawing = mp.solutions.drawing_utils
    frame_display = st.empty()

    recording = st.session_state.recording
    recorded_data = {
        "left_xs": [], "right_xs": [],
        "left_ys": [], "right_ys": [],
        "frame_idx": 0,
        "joints": {
            'LEFT_KNEE': [], 'RIGHT_KNEE': [],
            'LEFT_HIP': [], 'RIGHT_HIP': [],
            'LEFT_ANKLE': [], 'RIGHT_ANKLE': []
        },
        "angles": {
            'left_knee': [], 'right_knee': [],
            'left_hip': [], 'right_hip': [],
            'left_ankle': [], 'right_ankle': []
        }
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_file = f"outputs/gait_live_{timestamp}.mp4"
    csv_file_path = f"outputs/gait_live_{timestamp}.csv"
    os.makedirs("outputs", exist_ok=True)
    out = cv2.VideoWriter(video_file, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
    csv_file = open(csv_file_path, 'w', newline='') if start else None
    csv_writer = csv.writer(csv_file) if csv_file else None

    if start:
        st.session_state.recording = True
        landmark_names = [l.name for l in mp_pose.PoseLandmark]
        header = ['frame'] + [f"{n}_{a}" for n in landmark_names for a in ['x', 'y', 'z', 'visibility']]
        csv_writer.writerow(header)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if zoom > 1.0:
            cx, cy = width // 2, height // 2
            nw, nh = int(width / zoom), int(height / zoom)
            left, top = cx - nw // 2, cy - nh // 2
            frame = frame[top:top+nh, left:left+nw]
            frame = cv2.resize(frame, (width, height))

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)
        if result.pose_landmarks:
            mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        frame_display.image(frame, channels="BGR", use_container_width=True)

        if st.session_state.recording:
            out.write(frame)
            lm = result.pose_landmarks.landmark if result.pose_landmarks else None
            if lm:
                row = [recorded_data["frame_idx"]] + [v for l in lm for v in (l.x, l.y, l.z, l.visibility)]
                csv_writer.writerow(row)

                recorded_data["left_xs"].append(lm[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x)
                recorded_data["right_xs"].append(lm[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].x)
                recorded_data["left_ys"].append(lm[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y)
                recorded_data["right_ys"].append(lm[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y)

                for joint in recorded_data["joints"]:
                    recorded_data["joints"][joint].append([
                        lm[mp_pose.PoseLandmark[joint]].x,
                        lm[mp_pose.PoseLandmark[joint]].y,
                        lm[mp_pose.PoseLandmark[joint]].z
                    ])

                def add_angle(key, a, b, c):
                    recorded_data["angles"][key].append(calculate_angle(lm[a], lm[b], lm[c]))

                add_angle("left_knee", mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE)
                add_angle("right_knee", mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE)
                add_angle("left_hip", mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE)
                add_angle("right_hip", mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE)
                add_angle("left_ankle", mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.LEFT_HEEL)
                add_angle("right_ankle", mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE, mp_pose.PoseLandmark.RIGHT_HEEL)

                recorded_data["frame_idx"] += 1

        if stop:
            st.session_state.recording = False
            break

    cap.release()
    if out: out.release()
    if csv_file: csv_file.close()

    if recorded_data["frame_idx"] > 0:
        st.session_state.recorded_data = recorded_data
        st.session_state.recorded_data["csv_path"] = csv_file_path
        st.session_state.recorded_data["video_path"] = video_file
        st.success("✅ Recording complete. Gait features will now be shown below.")

        rd = st.session_state.recorded_data
        fps = 30
        duration = rd["frame_idx"] / fps
        foot_dists = np.sqrt((np.array(rd["left_xs"]) - np.array(rd["right_xs"]))**2 + (np.array(rd["left_ys"]) - np.array(rd["right_ys"]))**2)
        peaks, _ = find_peaks(foot_dists, distance=5)
        num_steps = len(peaks)
        cadence = (num_steps / duration) * 60 if duration > 0 else 0
        step_time = duration / num_steps if num_steps > 0 else 0
        step_lengths = foot_dists[peaks]
        mean_step_length = np.mean(step_lengths) if len(step_lengths) > 0 else 0
        mean_step_width = np.mean(np.abs(np.array(rd["left_ys"]) - np.array(rd["right_ys"])))
        stride_length = 2 * mean_step_length
        gait_speed = stride_length / (2 * step_time) if step_time > 0 else 0

        st.subheader("📊 Gait Characteristics")
        df_metrics = pd.DataFrame({
            "Metric": ["Cadence", "Step Time", "Step Length", "Step Width", "Stride Length", "Gait Speed", "Gait Cycle Duration"],
            "Value": [f"{cadence:.2f} steps/min", f"{step_time:.2f} s", f"{mean_step_length:.2f} m", f"{mean_step_width:.2f} m", f"{stride_length:.2f} m", f"{gait_speed:.2f} m/s", f"{duration:.2f} s"]
        })
        st.dataframe(df_metrics, use_container_width=True)

        st.subheader("🦵 Joint ROM (Degrees)")
        for joint, angles in rd["angles"].items():
            if angles:
                rom = max(angles) - min(angles)
                st.markdown(f"- **{joint.replace('_', ' ').title()} ROM:** `{rom:.2f}°`")

        st.subheader("📈 Step Distance Signal")
        fig, ax = plt.subplots()
        ax.plot(foot_dists, label='Foot Distance Signal')
        ax.plot(peaks, foot_dists[peaks], "rx", label='Detected Steps')
        ax.set_title("Gait Step Detection")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Normalized Distance")
        ax.legend()
        st.pyplot(fig)

        st.info(f"📄 CSV: `{rd['csv_path']}`\n🎥 Video: `{rd['video_path']}`")