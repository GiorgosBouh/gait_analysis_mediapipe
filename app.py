import streamlit as st
import os
from gait_live import run_live_gait_analysis
from gait_from_video import run_video_gait_analysis

st.title("Gait Analysis with MediaPipe")

mode = st.radio("Select input source:", ["Live Camera", "Video from GitHub folder"])

if mode == "Live Camera":
    st.write("Starting live gait analysis using your webcam...")
    if st.button("Start Live Capture"):
        run_live_gait_analysis()

elif mode == "Video from GitHub folder":
    video_files = [f for f in os.listdir("videos") if f.endswith(".mp4")]
    selected_video = st.selectbox("Select a video:", video_files)

    if st.button("Run analysis on selected video"):
        run_video_gait_analysis(f"videos/{selected_video}")