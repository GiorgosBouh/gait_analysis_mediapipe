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

def run_live_gait_analysis():
    st.title("🎥 Live Gait Camera Preview")
    st.markdown("Adjust your position and framing. Then press **Start Recording** when ready.")

    if "recording" not in st.session_state:
        st.session_state.recording = False

    zoom = st.slider("Zoom level (simulated crop)", 1.0, 2.0, 1.0, step=0.1)

    if not st.session_state.recording:
        if st.button("▶️ Start Recording"):
            st.session_state.recording = True
    else:
        if st.button("⏹️ Stop Recording"):
            st.session_state.recording = False

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("❌ Cannot open webcam. Make sure it is connected and accessible.")
        return

    width, height = int(cap.get(3)), int(cap.get(4))
    frame_display = st.empty()
    writer = None
    csv_writer = None
    csv_file = None
    out = None
    frame_idx = 0
    fps = 30

    left_xs, right_xs = [], []
    left_ys, right_ys = [], []

    joints = {
        'LEFT_KNEE': [], 'RIGHT_KNEE': [],
        'LEFT_HIP': [], 'RIGHT_HIP': [],
        'LEFT_ANKLE': [], 'RIGHT_ANKLE': []
    }

    recording = False

    while cap.isOpened():
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

        if st.session_state.recording and not recording:
            recording = True
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
            st.success("🔴 Recording started...")
            frame_idx = 0

        if recording:
            out.write(frame)
            if result.pose_landmarks:
                row = [frame_idx]
                for lm in result.pose_landmarks.landmark:
                    row.extend([lm.x, lm.y, lm.z, lm.visibility])
                csv_writer.writerow(row)

                left_xs.append(result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x)
                left_ys.append(result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y)
                right_xs.append(result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].x)
                right_ys.append(result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y)

                for joint in joints:
                    lm = result.pose_landmarks.landmark[mp_pose.PoseLandmark[joint]]
                    joints[joint].append([lm.x, lm.y, lm.z])

            frame_idx += 1

        if not st.session_state.recording and recording:
            recording = False
            out.release()
            csv_file.close()

            st.success("✅ Recording complete.")
            st.write(f"📄 CSV saved to: `{csv_file_path}`")
            st.write(f"🎥 Video saved to: `{video_file}`")

            duration = frame_idx / fps
            foot_dists = np.sqrt((np.array(left_xs) - np.array(right_xs))**2 + (np.array(left_ys) - np.array(right_ys))**2)
            peaks, _ = find_peaks(foot_dists, distance=5)
            num_steps = len(peaks)
            cadence = (num_steps / duration) * 60 if duration > 0 else 0
            step_time = duration / num_steps if num_steps > 0 else 0
            step_lengths = foot_dists[peaks]
            mean_step_length = np.mean(step_lengths) if len(step_lengths) > 0 else 0
            mean_step_width = np.mean(np.abs(np.array(left_ys) - np.array(right_ys)))
            stride_length = 2 * mean_step_length
            gait_speed = stride_length / (2 * step_time) if step_time > 0 else 0

            st.subheader("📊 Gait Characteristics")
            data = {
                "Cadence (steps/min)": f"{cadence:.2f}",
                "Step Time (s)": f"{step_time:.2f}",
                "Step Length": f"{mean_step_length:.2f}",
                "Step Width": f"{mean_step_width:.2f}",
                "Stride Length": f"{stride_length:.2f}",
                "Gait Speed (unit/s)": f"{gait_speed:.2f}",
                "Gait Cycle Duration (s)": f"{duration:.2f}"
            }
            st.dataframe(pd.DataFrame.from_dict(data, orient='index', columns=['Value']))

            st.subheader("🦵 Joint Range of Motion (3D)")
            for joint, coords in joints.items():
                arr = np.array(coords)
                min_vals = np.min(arr, axis=0)
                max_vals = np.max(arr, axis=0)
                rom = max_vals - min_vals
                st.markdown(f"- **{joint.replace('_', ' ').title()} ROM:** `x: {rom[0]:.3f}`, `y: {rom[1]:.3f}`, `z: {rom[2]:.3f}`")

            # Optional chart
            st.subheader("📈 Step Distance Signal")
            fig, ax = plt.subplots()
            ax.plot(foot_dists, label='Foot Distance Signal')
            ax.plot(peaks, foot_dists[peaks], "rx", label='Detected Steps')
            ax.set_title("Gait Step Detection")
            ax.set_xlabel("Frame")
            ax.set_ylabel("Normalized Distance")
            ax.legend()
            st.pyplot(fig)
            break

    cap.release()