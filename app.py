# --- app.py ---
import streamlit as st
import os
from gait_live import run_live_gait_analysis
from gait_from_video import run_video_gait_analysis

st.set_page_config(page_title="Gait Analysis", layout="centered")

st.title("Gait Analysis with MediaPipe")
st.markdown("Analyze human gait using pose landmarks extracted from video or webcam input.")

mode = st.radio("Select input source:", ["Live Camera", "Video from GitHub folder"])

# Track mode state
if "mode_selected" not in st.session_state:
    st.session_state.mode_selected = None

if mode == "Live Camera":
    st.write("Start live gait analysis using your connected webcam.")
    if st.session_state.mode_selected != "live":
        if st.button("Start Live Capture"):
            st.session_state.mode_selected = "live"
            st.experimental_rerun()

elif mode == "Video from GitHub folder":
    video_files = [f for f in os.listdir("videos") if f.endswith(".mp4")]
    if not video_files:
        st.warning("No .mp4 files found in the 'videos/' folder.")
    else:
        selected_video = st.selectbox("Select a video to analyze:", video_files)
        if st.button("Run analysis on selected video"):
            st.session_state.mode_selected = f"video::{selected_video}"
            st.experimental_rerun()

# Launch appropriate mode
if st.session_state.mode_selected == "live":
    run_live_gait_analysis()
elif isinstance(st.session_state.mode_selected, str) and st.session_state.mode_selected.startswith("video::"):
    selected = st.session_state.mode_selected.split("::")[1]
    run_video_gait_analysis(f"videos/{selected}")

# --- Footer ---
st.markdown("---")
st.markdown("\u00a9 2025 • Implemented by **Dr. Georgios Bouchouras**")