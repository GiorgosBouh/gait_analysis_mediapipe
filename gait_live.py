import cv2
import mediapipe as mp
import os
import csv
from datetime import datetime
import streamlit as st


def run_live_gait_analysis():
    st.title("🎥 Live Gait Camera Preview")
    st.markdown("Adjust your position and framing. Then press **Start Recording** when ready.")

    # UI controls
    zoom = st.slider("Zoom level (simulated crop)", 1.0, 2.0, 1.0, step=0.1)
    start_recording = st.button("▶️ Start Recording")
    stop_recording = st.button("⏹️ Stop Recording")

    # MediaPipe setup
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mp_drawing = mp.solutions.drawing_utils

    # Setup video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("❌ Cannot open webcam. Make sure it is connected and accessible.")
        return

    width, height = int(cap.get(3)), int(cap.get(4))

    # Initialize placeholders
    frame_display = st.empty()
    recording = False
    writer = None
    csv_writer = None
    csv_file = None
    out = None
    frame_idx = 0

    # Main loop
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Simulated zoom: crop and resize
        if zoom > 1.0:
            center_x, center_y = width // 2, height // 2
            new_w, new_h = int(width / zoom), int(height / zoom)
            left, top = center_x - new_w // 2, center_y - new_h // 2
            frame = frame[top:top + new_h, left:left + new_w]
            frame = cv2.resize(frame, (width, height))

        # Pose detection
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)
        if result.pose_landmarks:
            mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Display live frame
        frame_display.image(frame, channels="BGR", use_container_width=True)

        # Start recording logic
        if start_recording and not recording:
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

        # During recording
        if recording:
            out.write(frame)
            if result.pose_landmarks:
                row = [frame_idx] + [getattr(lm, attr) for lm in result.pose_landmarks.landmark for attr in ['x', 'y', 'z', 'visibility']]
                csv_writer.writerow(row)
            frame_idx += 1

        # Stop recording logic
        if recording and stop_recording:
            recording = False
            out.release()
            csv_file.close()
            st.success("✅ Recording complete.")
            st.write(f"📄 CSV saved to: `{csv_file_path}`")
            st.write(f"🎥 Video saved to: `{video_file}`")
            break

    cap.release()