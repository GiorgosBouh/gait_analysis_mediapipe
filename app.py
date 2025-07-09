import streamlit as st
import os
from gait_live import run_live_gait_analysis
from gait_from_video import run_video_gait_analysis

st.set_page_config(page_title="Gait Analysis", layout="centered")

st.title("Gait Analysis with MediaPipe")
st.markdown("Analyze human gait using pose landmarks extracted from video or webcam input.")

mode = st.radio("Select input source:", ["Live Camera", "Video from GitHub folder"])

if mode == "Live Camera":
    st.write("Start live gait analysis using your connected webcam.")
    if st.button("Start Live Capture"):
        run_live_gait_analysis()

elif mode == "Video from GitHub folder":
    video_files = [f for f in os.listdir("videos") if f.endswith(".mp4")]
    if not video_files:
        st.warning("No .mp4 files found in the 'videos/' folder.")
    else:
        selected_video = st.selectbox("Select a video to analyze:", video_files)
        if st.button("Run analysis on selected video"):
            run_video_gait_analysis(f"videos/{selected_video}")

# --- Footer ---
st.markdown("---")
st.markdown("© 2025 • Implemented by **Dr. Georgios Bouchouras**")