import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import csv
import os
from datetime import datetime

# Setup
st.title("🧠 MediaPipe Face Mesh – Live with Recording")
st.markdown("Live detection of **468 facial landmarks** using your webcam, with optional recording.")

# MediaPipe init
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# State variables
if "camera_running" not in st.session_state:
    st.session_state.camera_running = False
if "recording" not in st.session_state:
    st.session_state.recording = False

# UI
col1, col2 = st.columns(2)
with col1:
    mirror = st.checkbox("🔄 Mirror view", value=True)
with col2:
    zoom = st.slider("🔍 Zoom level", 1.0, 2.0, 1.0, 0.1)

# Controls
start_camera = st.button("📷 Start Camera")
stop_camera = st.button("❌ Stop Camera")
start_record = st.button("🔴 Start Recording")
stop_record = st.button("⏹️ Stop Recording")

FRAME_WINDOW = st.image([])
save_folder = "outputs"
os.makedirs(save_folder, exist_ok=True)

if start_camera:
    st.session_state.camera_running = True

if stop_camera:
    st.session_state.camera_running = False
    st.session_state.recording = False  # force stop recording too

# Camera session
if st.session_state.camera_running:
    cap = cv2.VideoCapture(0)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(save_folder, f"face_mesh_{timestamp}.csv")
    video_path = os.path.join(save_folder, f"face_mesh_{timestamp}.mp4")

    width = int(cap.get(3))
    height = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None
    csv_file = None
    writer = None

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        frame_idx = 0
        while st.session_state.camera_running:
            ret, frame = cap.read()
            if not ret:
                st.warning("⚠️ Could not read frame from camera.")
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

                    if st.session_state.recording and writer:
                        for idx, landmark in enumerate(face_landmarks.landmark):
                            writer.writerow([frame_idx, idx, landmark.x, landmark.y, landmark.z])

            if st.session_state.recording and out:
                out.write(frame)

            FRAME_WINDOW.image(frame, channels="BGR", use_container_width=True)
            frame_idx += 1

            if start_record and not st.session_state.recording:
                st.session_state.recording = True
                out = cv2.VideoWriter(video_path, fourcc, 30.0, (width, height))
                csv_file = open(csv_path, mode='w', newline='')
                writer = csv.writer(csv_file)
                writer.writerow(['frame', 'landmark_id', 'x', 'y', 'z'])
                st.toast("🔴 Recording started")

            if stop_record and st.session_state.recording:
                st.session_state.recording = False
                if out:
                    out.release()
                if csv_file:
                    csv_file.close()
                st.toast("⏹️ Recording stopped")
                st.success(f"✅ Files saved:\n- 📄 CSV: `{csv_path}`\n- 🎥 Video: `{video_path}`")
                out = None
                writer = None
                csv_file = None

    cap.release()
    st.info("📷 Camera session ended.")