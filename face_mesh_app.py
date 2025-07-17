import streamlit as st
import cv2
import mediapipe as mp
import numpy as np

# Setup
st.title("🧠 MediaPipe Face Mesh – Live Demo")
st.markdown("Live detection of **468 facial landmarks** using your webcam.")

# MediaPipe initialization
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Checkbox for mirror mode
mirror = st.checkbox("🔄 Mirror view", value=True)
zoom = st.slider("🔍 Zoom level", 1.0, 2.0, 1.0, 0.1)

# Start button
run = st.checkbox("▶️ Start camera")

FRAME_WINDOW = st.image([])

if run:
    cap = cv2.VideoCapture(0)

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        
        width = int(cap.get(3))
        height = int(cap.get(4))

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

            FRAME_WINDOW.image(frame, channels="BGR", use_container_width=True)

        cap.release()
else:
    st.info("Enable 'Start camera' to begin.")