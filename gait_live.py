import cv2
import mediapipe as mp
import os
import csv
from datetime import datetime
import streamlit as st

def run_live_gait_analysis():
    os.makedirs("outputs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = f"outputs/gait_live_{timestamp}.csv"
    video_file = f"outputs/gait_live_{timestamp}.mp4"

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    width, height = int(cap.get(3)), int(cap.get(4))
    out = cv2.VideoWriter(video_file, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

    landmark_names = [l.name for l in mp_pose.PoseLandmark]
    header = ['frame'] + [f"{n}_{a}" for n in landmark_names for a in ['x', 'y', 'z', 'visibility']]
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        frame_idx = 0
        frame_display = st.empty()
        stop = st.button("Stop Recording")

        while cap.isOpened() and not stop:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)
            if result.pose_landmarks:
                mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                row = [frame_idx] + [getattr(lm, attr) for lm in result.pose_landmarks.landmark for attr in ['x', 'y', 'z', 'visibility']]
                writer.writerow(row)
            out.write(frame)
            frame_display.image(frame, channels="BGR", caption=f"Frame {frame_idx}", use_column_width=True)
            frame_idx += 1

    cap.release()
    out.release()
    st.success("✅ Recording complete.")
    st.write(f"📄 CSV saved to: `{csv_file}`")
    st.write(f"🎥 Video saved to: `{video_file}`")