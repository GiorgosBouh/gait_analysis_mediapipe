import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import csv
import os
from datetime import datetime

# Setup
st.title("🧠 MediaPipe Face Mesh – Live with Recording")
st.markdown("Live detection of **468 facial landmarks** using your webcam, with data recording and export.")

# MediaPipe initialization
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Controls
mirror = st.checkbox("🔄 Mirror view", value=True)
zoom = st.slider("🔍 Zoom level", 1.0, 2.0, 1.0, 0.1)
run = st.checkbox("▶️ Start camera and record")

FRAME_WINDOW = st.image([])
save_folder = "outputs"
os.makedirs(save_folder, exist_ok=True)

if run:
    cap = cv2.VideoCapture(0)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(save_folder, f"face_mesh_{timestamp}.csv")
    video_path = os.path.join(save_folder, f"face_mesh_{timestamp}.mp4")

    width = int(cap.get(3))
    height = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 30.0, (width, height))

    # Prepare CSV
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame', 'landmark_id', 'x', 'y', 'z'])

        with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as face_mesh:

            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    st.warning("⚠️ Could not access camera.")
                    break

                if mirror:
                    frame = cv2.flip(frame, 1)

                if zoom > 1.0:
                    center_x, center_y = width // 2, height // 2
                    new_w, new_h = int(width / zoom), int(height / zoom)
                    left, top = center_x - new_w // 2, center_y - new_h // 2
                    frame = frame[top:top + new_h, left:left + new_w]
                    frame = cv2.resize(frame, (width, height))

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb)

                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        mp_drawing.draw_landmarks(
                            image=frame,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=drawing_spec,
                            connection_drawing_spec=drawing_spec
                        )
                        for idx, landmark in enumerate(face_landmarks.landmark):
                            writer.writerow([frame_idx, idx, landmark.x, landmark.y, landmark.z])

                out.write(frame)
                FRAME_WINDOW.image(frame, channels="BGR", use_container_width=True)
                frame_idx += 1

    cap.release()
    out.release()
    st.success("✅ Recording finished.")
    st.markdown(f"- 📄 CSV saved to: `{csv_path}`")
    st.markdown(f"- 🎥 Video saved to: `{video_path}`")
else:
    st.info("Enable 'Start camera and record' to begin.")