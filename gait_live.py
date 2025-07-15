import cv2
import mediapipe as mp
import os
import csv
from datetime import datetime
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from math import acos, degrees
from collections import defaultdict
import threading
import time
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, VideoProcessorBase
import av

# Constants
PLOT_WIDTH = 12
PLOT_HEIGHT = 6
SMOOTH_WINDOW = 5  # for smoothing angle data

# RTC Configuration for WebRTC
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

def calculate_angle(a, b, c):
    """Calculate 3D angle between three points"""
    a = np.array([a.x, a.y, a.z])
    b = np.array([b.x, b.y, b.z])
    c = np.array([c.x, c.y, c.z])
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = degrees(acos(np.clip(cosine_angle, -1.0, 1.0)))
    return angle

def smooth_data(data, window_size=SMOOTH_WINDOW):
    """Apply simple moving average smoothing"""
    return savgol_filter(data, window_size, 2) if len(data) > window_size else data

def detect_gait_phases(foot_distances, prominence=0.05):
    """Detect gait phases (stance/swing) from foot distances"""
    peaks, _ = find_peaks(foot_distances, prominence=prominence)
    valleys, _ = find_peaks(-foot_distances, prominence=prominence)
    
    # Create gait phase segments
    phases = []
    for i in range(len(peaks) - 1):
        start = peaks[i]
        # Find the first valley between two peaks
        mid_indices = np.where((valleys > peaks[i]) & (valleys < peaks[i + 1]))[0]
        if len(mid_indices) > 0:
            mid = valleys[mid_indices[0]]
            end = peaks[i + 1]
            phases.append(('stance', start, mid))
            phases.append(('swing', mid, end))
    return phases

class GaitAnalysisVideoProcessor(VideoProcessorBase):
    """Video processor for gait analysis using MediaPipe"""
    
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.lock = threading.Lock()
        
        # Data storage
        self.recorded_data = {
            "left_foot": {"x": [], "y": [], "z": []},
            "right_foot": {"x": [], "y": [], "z": []},
            "joint_positions": defaultdict(list),
            "joint_angles": defaultdict(list),
            "frame_count": 0,
            "recording": False,
            "frames": []
        }
        
    def start_recording(self):
        """Start recording gait data"""
        with self.lock:
            self.recorded_data = {
                "left_foot": {"x": [], "y": [], "z": []},
                "right_foot": {"x": [], "y": [], "z": []},
                "joint_positions": defaultdict(list),
                "joint_angles": defaultdict(list),
                "frame_count": 0,
                "recording": True,
                "frames": []
            }
    
    def stop_recording(self):
        """Stop recording gait data"""
        with self.lock:
            self.recorded_data["recording"] = False
    
    def is_recording(self):
        """Check if currently recording"""
        with self.lock:
            return self.recorded_data["recording"]
    
    def get_recorded_data(self):
        """Get recorded data"""
        with self.lock:
            return self.recorded_data.copy()
    
    def recv(self, frame):
        """Process each frame from the webcam"""
        img = frame.to_ndarray(format="bgr24")
        
        # Flip image horizontally for mirror effect
        img = cv2.flip(img, 1)
        
        # Convert BGR to RGB for MediaPipe
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.pose.process(rgb_img)
        
        # Draw pose landmarks
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                img, 
                results.pose_landmarks, 
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )
            
            # Record data if recording is active
            with self.lock:
                if self.recorded_data["recording"]:
                    self._record_frame_data(results.pose_landmarks.landmark, img)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")
    
    def _record_frame_data(self, landmarks, frame):
        """Record frame data during recording"""
        lm = landmarks
        rd = self.recorded_data
        
        # Store frame
        rd["frames"].append(frame.copy())
        
        # Store foot positions
        rd["left_foot"]["x"].append(lm[self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x)
        rd["left_foot"]["y"].append(lm[self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y)
        rd["left_foot"]["z"].append(lm[self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX].z)
        
        rd["right_foot"]["x"].append(lm[self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].x)
        rd["right_foot"]["y"].append(lm[self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y)
        rd["right_foot"]["z"].append(lm[self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].z)

        # Store joint positions
        for joint in ['LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_HIP', 'RIGHT_HIP', 'LEFT_ANKLE', 'RIGHT_ANKLE']:
            rd["joint_positions"][joint].append([
                lm[self.mp_pose.PoseLandmark[joint]].x,
                lm[self.mp_pose.PoseLandmark[joint]].y,
                lm[self.mp_pose.PoseLandmark[joint]].z
            ])

        # Calculate and store joint angles
        rd["joint_angles"]["left_knee"].append(
            calculate_angle(lm[self.mp_pose.PoseLandmark.LEFT_HIP], 
                          lm[self.mp_pose.PoseLandmark.LEFT_KNEE], 
                          lm[self.mp_pose.PoseLandmark.LEFT_ANKLE])
        )
        rd["joint_angles"]["right_knee"].append(
            calculate_angle(lm[self.mp_pose.PoseLandmark.RIGHT_HIP], 
                           lm[self.mp_pose.PoseLandmark.RIGHT_KNEE], 
                           lm[self.mp_pose.PoseLandmark.RIGHT_ANKLE])
        )
        rd["joint_angles"]["left_hip"].append(
            calculate_angle(lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER], 
                          lm[self.mp_pose.PoseLandmark.LEFT_HIP], 
                          lm[self.mp_pose.PoseLandmark.LEFT_KNEE])
        )
        rd["joint_angles"]["right_hip"].append(
            calculate_angle(lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER], 
                          lm[self.mp_pose.PoseLandmark.RIGHT_HIP], 
                          lm[self.mp_pose.PoseLandmark.RIGHT_KNEE])
        )
        rd["joint_angles"]["left_ankle"].append(
            calculate_angle(lm[self.mp_pose.PoseLandmark.LEFT_KNEE], 
                          lm[self.mp_pose.PoseLandmark.LEFT_ANKLE], 
                          lm[self.mp_pose.PoseLandmark.LEFT_HEEL])
        )
        rd["joint_angles"]["right_ankle"].append(
            calculate_angle(lm[self.mp_pose.PoseLandmark.RIGHT_KNEE], 
                           lm[self.mp_pose.PoseLandmark.RIGHT_ANKLE], 
                           lm[self.mp_pose.PoseLandmark.RIGHT_HEEL])
        )

        rd["frame_count"] += 1

def run_live_gait_analysis():
    st.title("🎥 Live Gait Analysis")
    st.markdown("""
    **Instructions:**
    1. Position yourself in the camera view
    2. Press **Start Recording** and walk naturally
    3. Press **Stop Recording** when finished
    4. View your gait analysis results
    """)

    # Initialize session state
    if "processor" not in st.session_state:
        st.session_state.processor = GaitAnalysisVideoProcessor()
    
    # WebRTC streamer
    webrtc_ctx = webrtc_streamer(
        key="gait-analysis",
        video_processor_factory=lambda: st.session_state.processor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    # Recording controls
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("▶️ Start Recording", type="primary"):
            if webrtc_ctx.video_processor:
                webrtc_ctx.video_processor.start_recording()
                st.success("🔴 Recording started!")
    
    with col2:
        if st.button("⏹️ Stop Recording", type="secondary"):
            if webrtc_ctx.video_processor:
                webrtc_ctx.video_processor.stop_recording()
                st.success("⏹️ Recording stopped!")
    
    with col3:
        if webrtc_ctx.video_processor:
            if webrtc_ctx.video_processor.is_recording():
                st.markdown("🔴 **RECORDING...**")
            else:
                st.markdown("⚪ **Ready**")

    # Display results if we have recorded data
    if webrtc_ctx.video_processor and not webrtc_ctx.video_processor.is_recording():
        recorded_data = webrtc_ctx.video_processor.get_recorded_data()
        
        if recorded_data["frame_count"] > 0:
            st.markdown("---")
            
            # Save data button
            if st.button("💾 Save Data & Show Analysis", type="primary"):
                save_recording_data(recorded_data)
                display_gait_analysis_results(recorded_data)

def save_recording_data(recorded_data):
    """Save recorded data to files"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("outputs", exist_ok=True)
    
    # Save CSV data
    csv_path = f"outputs/gait_live_{timestamp}.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(['frame', 'left_foot_x', 'left_foot_y', 'left_foot_z',
                        'right_foot_x', 'right_foot_y', 'right_foot_z'] +
                       [f"{joint}_{coord}" for joint in recorded_data["joint_positions"].keys() 
                        for coord in ['x', 'y', 'z']] +
                       list(recorded_data["joint_angles"].keys()))
        
        # Write data row by row
        for i in range(recorded_data["frame_count"]):
            row = [i]
            row.extend([recorded_data["left_foot"]["x"][i], 
                       recorded_data["left_foot"]["y"][i], 
                       recorded_data["left_foot"]["z"][i]])
            row.extend([recorded_data["right_foot"]["x"][i], 
                       recorded_data["right_foot"]["y"][i], 
                       recorded_data["right_foot"]["z"][i]])
            
            for joint in recorded_data["joint_positions"].values():
                row.extend(joint[i])
                
            for angle in recorded_data["joint_angles"].values():
                row.append(angle[i])
                
            writer.writerow(row)
    
    # Save video
    video_path = f"outputs/gait_live_{timestamp}.mp4"
    if recorded_data["frames"]:
        height, width, layers = recorded_data["frames"][0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 30.0, (width, height))
        
        for frame in recorded_data["frames"]:
            out.write(frame)
        
        out.release()
    
    st.success(f"✅ Data saved successfully!")
    st.info(f"📁 CSV: {csv_path}")
    st.info(f"🎥 Video: {video_path}")

def display_gait_analysis_results(recorded_data):
    """Display all gait analysis results after recording"""
    st.success("✅ Recording complete. Gait analysis results:")
    st.markdown("---")
    
    # Basic metrics
    fps = 30  # Assuming 30 FPS
    duration = recorded_data["frame_count"] / fps
    left_foot_x = np.array(recorded_data["left_foot"]["x"])
    left_foot_y = np.array(recorded_data["left_foot"]["y"])
    right_foot_x = np.array(recorded_data["right_foot"]["x"])
    right_foot_y = np.array(recorded_data["right_foot"]["y"])
    
    # Calculate step distances and detect gait phases
    foot_dists = np.sqrt((left_foot_x - right_foot_x)**2 + (left_foot_y - right_foot_y)**2)
    gait_phases = detect_gait_phases(foot_dists)
    peaks, _ = find_peaks(foot_dists, prominence=0.05)
    num_steps = len(peaks)
    
    # Calculate ROM for each gait phase
    phase_rom = defaultdict(list)
    for phase in gait_phases:
        phase_name, start, end = phase
        for joint, angles in recorded_data["joint_angles"].items():
            phase_angles = angles[start:end]
            if phase_angles:
                rom = max(phase_angles) - min(phase_angles)
                phase_rom[f"{joint}_{phase_name}"].append(rom)
    
    # Calculate mean ROM per phase
    mean_phase_rom = {}
    for key, values in phase_rom.items():
        mean_phase_rom[key] = np.mean(values) if values else 0
    
    # Gait metrics - separate for left and right
    cadence = (num_steps / duration) * 60 if duration > 0 else 0
    step_time = duration / num_steps if num_steps > 0 else 0
    
    # Left side metrics
    left_step_lengths = [foot_dists[peaks[i]] for i in range(len(peaks)) if i % 2 == 0]
    left_mean_step_length = np.mean(left_step_lengths) if left_step_lengths else 0
    left_stride_length = 2 * left_mean_step_length
    
    # Right side metrics
    right_step_lengths = [foot_dists[peaks[i]] for i in range(len(peaks)) if i % 2 == 1]
    right_mean_step_length = np.mean(right_step_lengths) if right_step_lengths else 0
    right_stride_length = 2 * right_mean_step_length
    
    mean_step_width = np.mean(np.abs(left_foot_y - right_foot_y))
    gait_speed = (left_stride_length + right_stride_length) / (4 * step_time) if step_time > 0 else 0
    
    # Display metrics
    st.subheader("📊 Gait Characteristics")
    metrics = {
        "Recording Duration": f"{duration:.2f} s",
        "Number of Steps": f"{num_steps} ({len(left_step_lengths)} left, {len(right_step_lengths)} right)",
        "Cadence": f"{cadence:.2f} steps/min",
        "Step Time": f"{step_time:.2f} s",
        "Left Step Length": f"{left_mean_step_length:.2f} (normalized)",
        "Right Step Length": f"{right_mean_step_length:.2f} (normalized)",
        "Left Stride Length": f"{left_stride_length:.2f} (normalized)",
        "Right Stride Length": f"{right_stride_length:.2f} (normalized)",
        "Step Width": f"{mean_step_width:.2f} (normalized)",
        "Gait Speed": f"{gait_speed:.2f} (normalized units/s)"
    }
    
    st.table(pd.DataFrame.from_dict(metrics, orient='index', columns=['Value']))
    
    # Joint ROM by phase
    st.subheader("🦵 Joint Range of Motion (ROM) by Gait Phase")
    rom_data = []
    for joint in recorded_data["joint_angles"].keys():
        rom_data.append({
            "Joint": joint.replace('_', ' ').title(),
            "Stance ROM (°)": f"{mean_phase_rom.get(f'{joint}_stance', 0):.1f}",
            "Swing ROM (°)": f"{mean_phase_rom.get(f'{joint}_swing', 0):.1f}",
            "Total ROM (°)": f"{max(recorded_data['joint_angles'][joint]) - min(recorded_data['joint_angles'][joint]):.1f}"
        })
    st.table(pd.DataFrame(rom_data))
    
    # Visualization section
    st.subheader("📈 Gait Visualizations")
    
    # Foot distance plot with gait phases
    fig, ax = plt.subplots(figsize=(PLOT_WIDTH, PLOT_HEIGHT))
    ax.plot(foot_dists, label='Foot Distance')
    ax.plot(peaks, foot_dists[peaks], "rx", label='Detected Steps')
    
    # Color gait phases
    for phase in gait_phases:
        phase_name, start, end = phase
        color = 'lightgreen' if phase_name == 'stance' else 'lightblue'
        ax.axvspan(start, end, color=color, alpha=0.3, label=f'{phase_name.capitalize()} Phase' if start == 0 else "")
    
    ax.set_title("Step Detection with Gait Phases")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Normalized Distance")
    ax.legend()
    st.pyplot(fig)
    
    # Joint angles plots with gait phases
    fig, axes = plt.subplots(2, 2, figsize=(PLOT_WIDTH, PLOT_HEIGHT*2))
    axes = axes.flatten()
    
    for i, (joint, angles) in enumerate(recorded_data["joint_angles"].items()):
        if i >= len(axes):
            break
        smoothed = smooth_data(angles)
        axes[i].plot(smoothed, label='Angle')
        
        # Mark gait phases
        for phase in gait_phases:
            phase_name, start, end = phase
            color = 'green' if phase_name == 'stance' else 'blue'
            axes[i].axvspan(start, end, color=color, alpha=0.1)
        
        axes[i].set_title(f"{joint.replace('_', ' ').title()} Angle")
        axes[i].set_xlabel("Frame")
        axes[i].set_ylabel("Angle (°)")
    
    plt.tight_layout()
    st.pyplot(fig)