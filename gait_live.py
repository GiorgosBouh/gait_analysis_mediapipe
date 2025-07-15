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

# Constants
PLOT_WIDTH = 12
PLOT_HEIGHT = 6
SMOOTH_WINDOW = 5

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
    
    phases = []
    for i in range(len(peaks) - 1):
        start = peaks[i]
        mid_indices = np.where((valleys > peaks[i]) & (valleys < peaks[i + 1]))[0]
        if len(mid_indices) > 0:
            mid = valleys[mid_indices[0]]
            end = peaks[i + 1]
            phases.append(('stance', start, mid))
            phases.append(('swing', mid, end))
    return phases

def run_live_gait_analysis():
    st.title("🎥 Live Gait Analysis")
    st.markdown("""
    **Instructions:**
    1. Position yourself in the camera view (side view recommended)
    2. Click **Start Recording** and walk naturally for 10-15 seconds
    3. Click **Stop Recording** to finish and see analysis
    """)

    # Initialize session state
    if "recording" not in st.session_state:
        st.session_state.recording = False
    if "recorded_data" not in st.session_state:
        st.session_state.recorded_data = {
            "left_foot": {"x": [], "y": [], "z": []},
            "right_foot": {"x": [], "y": [], "z": []},
            "joint_positions": defaultdict(list),
            "joint_angles": defaultdict(list),
            "frame_count": 0,
            "csv_path": None,
            "video_path": None
        }

    # Camera controls
    col1, col2 = st.columns(2)
    with col1:
        zoom = st.slider("Zoom level", 1.0, 2.0, 1.0, 0.1)
    with col2:
        mirror = st.checkbox("Mirror view", True)

    # Recording controls
    col3, col4, col5 = st.columns(3)
    with col3:
        if not st.session_state.recording:
            if st.button("▶️ Start Recording", type="primary"):
                st.session_state.recording = True
                # Reset data
                st.session_state.recorded_data = {
                    "left_foot": {"x": [], "y": [], "z": []},
                    "right_foot": {"x": [], "y": [], "z": []},
                    "joint_positions": defaultdict(list),
                    "joint_angles": defaultdict(list),
                    "frame_count": 0,
                    "csv_path": None,
                    "video_path": None
                }
                st.rerun()

    with col4:
        if st.session_state.recording:
            if st.button("⏹️ Stop Recording", type="secondary"):
                st.session_state.recording = False
                st.rerun()

    with col5:
        if st.session_state.recording:
            st.markdown("🔴 **RECORDING...**")
        else:
            st.markdown("⚪ **Ready**")

    # Initialize MediaPipe
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    # Camera setup
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("❌ Cannot open webcam. Make sure it is connected and no other application is using it.")
        st.info("💡 Try closing other applications that might be using the camera (Chrome, Zoom, etc.)")
        return

    width, height = int(cap.get(3)), int(cap.get(4))
    frame_display = st.empty()
    status_display = st.empty()

    # Video writer setup for recording
    video_writer = None
    csv_writer = None
    csv_file = None
    
    if st.session_state.recording:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("outputs", exist_ok=True)
        video_path = f"outputs/gait_live_{timestamp}.mp4"
        csv_path = f"outputs/gait_live_{timestamp}.csv"
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_path, fourcc, 30.0, (width, height))
        
        # Setup CSV writer
        csv_file = open(csv_path, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        landmark_names = [l.name for l in mp_pose.PoseLandmark]
        header = ['frame'] + [f"{n}_{a}" for n in landmark_names for a in ['x', 'y', 'z', 'visibility']]
        csv_writer.writerow(header)
        
        st.session_state.recorded_data["csv_path"] = csv_path
        st.session_state.recorded_data["video_path"] = video_path

    # Main processing loop
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                st.warning("⚠️ Couldn't read frame from camera")
                break

            # Apply mirror effect if enabled
            if mirror:
                frame = cv2.flip(frame, 1)

            # Apply zoom if needed
            if zoom > 1.0:
                center_x, center_y = width // 2, height // 2
                new_w, new_h = int(width / zoom), int(height / zoom)
                left, top = center_x - new_w // 2, center_y - new_h // 2
                frame = frame[top:top + new_h, left:left + new_w]
                frame = cv2.resize(frame, (width, height))

            # Process frame with MediaPipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)

            # Draw landmarks if detected
            if result.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame, 
                    result.pose_landmarks, 
                    mp_pose.POSE_CONNECTIONS,
                    mp.solutions.drawing_utils.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                    mp.solutions.drawing_utils.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                )
                
                # Recording logic
                if st.session_state.recording and video_writer and csv_writer:
                    # Write frame to video
                    video_writer.write(frame)
                    
                    # Extract landmarks
                    lm = result.pose_landmarks.landmark
                    rd = st.session_state.recorded_data

                    # Store foot positions
                    rd["left_foot"]["x"].append(lm[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x)
                    rd["left_foot"]["y"].append(lm[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y)
                    rd["left_foot"]["z"].append(lm[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].z)
                    
                    rd["right_foot"]["x"].append(lm[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].x)
                    rd["right_foot"]["y"].append(lm[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y)
                    rd["right_foot"]["z"].append(lm[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].z)

                    # Store joint positions
                    for joint in ['LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_HIP', 'RIGHT_HIP', 'LEFT_ANKLE', 'RIGHT_ANKLE']:
                        rd["joint_positions"][joint].append([
                            lm[mp_pose.PoseLandmark[joint]].x,
                            lm[mp_pose.PoseLandmark[joint]].y,
                            lm[mp_pose.PoseLandmark[joint]].z
                        ])

                    # Calculate and store joint angles
                    rd["joint_angles"]["left_knee"].append(
                        calculate_angle(lm[mp_pose.PoseLandmark.LEFT_HIP], 
                                      lm[mp_pose.PoseLandmark.LEFT_KNEE], 
                                      lm[mp_pose.PoseLandmark.LEFT_ANKLE])
                    )
                    rd["joint_angles"]["right_knee"].append(
                        calculate_angle(lm[mp_pose.PoseLandmark.RIGHT_HIP], 
                                       lm[mp_pose.PoseLandmark.RIGHT_KNEE], 
                                       lm[mp_pose.PoseLandmark.RIGHT_ANKLE])
                    )
                    rd["joint_angles"]["left_hip"].append(
                        calculate_angle(lm[mp_pose.PoseLandmark.LEFT_SHOULDER], 
                                      lm[mp_pose.PoseLandmark.LEFT_HIP], 
                                      lm[mp_pose.PoseLandmark.LEFT_KNEE])
                    )
                    rd["joint_angles"]["right_hip"].append(
                        calculate_angle(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER], 
                                      lm[mp_pose.PoseLandmark.RIGHT_HIP], 
                                      lm[mp_pose.PoseLandmark.RIGHT_KNEE])
                    )
                    rd["joint_angles"]["left_ankle"].append(
                        calculate_angle(lm[mp_pose.PoseLandmark.LEFT_KNEE], 
                                      lm[mp_pose.PoseLandmark.LEFT_ANKLE], 
                                      lm[mp_pose.PoseLandmark.LEFT_HEEL])
                    )
                    rd["joint_angles"]["right_ankle"].append(
                        calculate_angle(lm[mp_pose.PoseLandmark.RIGHT_KNEE], 
                                       lm[mp_pose.PoseLandmark.RIGHT_ANKLE], 
                                       lm[mp_pose.PoseLandmark.RIGHT_HEEL])
                    )

                    # Save to CSV
                    row = [rd["frame_count"]] + [getattr(landmark, attr) for landmark in lm for attr in ['x', 'y', 'z', 'visibility']]
                    csv_writer.writerow(row)

                    rd["frame_count"] += 1
                    
                    # Update status
                    status_display.info(f"🔴 Recording... Frame: {rd['frame_count']}")
            
            elif st.session_state.recording:
                # Still recording but no pose detected
                if video_writer:
                    video_writer.write(frame)
                status_display.warning("⚠️ No pose detected - make sure you're visible in the camera")
            
            # Show frame
            frame_display.image(frame, channels="BGR", use_container_width=True)

            # Break loop when recording is stopped
            if not st.session_state.recording:
                break

    except Exception as e:
        st.error(f"Camera error: {str(e)}")
    
    finally:
        # Release resources
        cap.release()
        if video_writer:
            video_writer.release()
        if csv_file:
            csv_file.close()
        pose.close()

    # Display results if we have recorded data
    if not st.session_state.recording and st.session_state.recorded_data["frame_count"] > 0:
        st.success("✅ Recording complete!")
        display_gait_analysis_results(st.session_state.recorded_data)

def display_gait_analysis_results(recorded_data):
    """Display comprehensive gait analysis results"""
    st.markdown("---")
    st.subheader("📊 Gait Analysis Results")
    
    rd = recorded_data
    
    # Basic metrics
    fps = 30
    duration = rd["frame_count"] / fps
    left_foot_x = np.array(rd["left_foot"]["x"])
    left_foot_y = np.array(rd["left_foot"]["y"])
    right_foot_x = np.array(rd["right_foot"]["x"])
    right_foot_y = np.array(rd["right_foot"]["y"])
    
    # Calculate step distances and detect gait phases
    foot_dists = np.sqrt((left_foot_x - right_foot_x)**2 + (left_foot_y - right_foot_y)**2)
    gait_phases = detect_gait_phases(foot_dists)
    peaks, _ = find_peaks(foot_dists, prominence=0.02, distance=10)
    num_steps = len(peaks)
    
    # Calculate ROM for each gait phase
    phase_rom = defaultdict(list)
    for phase in gait_phases:
        phase_name, start, end = phase
        for joint, angles in rd["joint_angles"].items():
            if start < len(angles) and end <= len(angles):
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
    
    # Step lengths
    left_step_lengths = [foot_dists[peaks[i]] for i in range(len(peaks)) if i % 2 == 0]
    right_step_lengths = [foot_dists[peaks[i]] for i in range(len(peaks)) if i % 2 == 1]
    
    left_mean_step_length = np.mean(left_step_lengths) if left_step_lengths else 0
    right_mean_step_length = np.mean(right_step_lengths) if right_step_lengths else 0
    left_stride_length = 2 * left_mean_step_length
    right_stride_length = 2 * right_mean_step_length
    
    mean_step_width = np.mean(np.abs(left_foot_y - right_foot_y))
    gait_speed = (left_stride_length + right_stride_length) / (4 * step_time) if step_time > 0 else 0
    
    # Display summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Duration", f"{duration:.1f}s")
        st.metric("Steps Detected", f"{num_steps}")
    
    with col2:
        st.metric("Cadence", f"{cadence:.0f} steps/min")
        st.metric("Step Time", f"{step_time:.2f}s")
    
    with col3:
        st.metric("Left Steps", f"{len(left_step_lengths)}")
        st.metric("Right Steps", f"{len(right_step_lengths)}")
    
    with col4:
        st.metric("Step Symmetry", f"{min(left_mean_step_length, right_mean_step_length) / max(left_mean_step_length, right_mean_step_length) if max(left_mean_step_length, right_mean_step_length) > 0 else 1:.2f}")
        st.metric("Gait Speed", f"{gait_speed:.2f} (norm)")

    # Detailed metrics table
    st.subheader("📊 Detailed Gait Characteristics")
    metrics = {
        "Recording Duration": f"{duration:.2f} s",
        "Number of Steps": f"{num_steps} ({len(left_step_lengths)} left, {len(right_step_lengths)} right)",
        "Cadence": f"{cadence:.2f} steps/min",
        "Step Time": f"{step_time:.2f} s",
        "Left Step Length": f"{left_mean_step_length:.3f} (normalized)",
        "Right Step Length": f"{right_mean_step_length:.3f} (normalized)",
        "Left Stride Length": f"{left_stride_length:.3f} (normalized)",
        "Right Stride Length": f"{right_stride_length:.3f} (normalized)",
        "Step Width": f"{mean_step_width:.3f} (normalized)",
        "Gait Speed": f"{gait_speed:.3f} (normalized units/s)"
    }
    
    st.table(pd.DataFrame.from_dict(metrics, orient='index', columns=['Value']))
    
    # Joint ROM by phase
    st.subheader("🦵 Joint Range of Motion (ROM) by Gait Phase")
    if rd["joint_angles"]:
        rom_data = []
        for joint in rd["joint_angles"].keys():
            angles = rd["joint_angles"][joint]
            if angles:
                rom_data.append({
                    "Joint": joint.replace('_', ' ').title(),
                    "Stance ROM (°)": f"{mean_phase_rom.get(f'{joint}_stance', 0):.1f}",
                    "Swing ROM (°)": f"{mean_phase_rom.get(f'{joint}_swing', 0):.1f}",
                    "Total ROM (°)": f"{max(angles) - min(angles):.1f}",
                    "Mean Angle (°)": f"{np.mean(angles):.1f}"
                })
        
        if rom_data:
            st.table(pd.DataFrame(rom_data))
    
    # Visualization section
    st.subheader("📈 Gait Visualizations")
    
    # Foot distance plot with gait phases
    fig, ax = plt.subplots(figsize=(PLOT_WIDTH, PLOT_HEIGHT))
    ax.plot(foot_dists, label='Inter-foot Distance', linewidth=2)
    ax.plot(peaks, foot_dists[peaks], "ro", markersize=8, label=f'Detected Steps ({num_steps})')
    
    # Color gait phases
    for i, phase in enumerate(gait_phases[:6]):  # Show first few phases for clarity
        phase_name, start, end = phase
        color = 'lightgreen' if phase_name == 'stance' else 'lightblue'
        ax.axvspan(start, end, color=color, alpha=0.3, 
                  label=f'{phase_name.capitalize()} Phase' if i < 2 else "")
    
    ax.set_title("Step Detection with Gait Phases", fontsize=16)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Normalized Distance")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    
    # Joint angles plots with gait phases
    if rd["joint_angles"]:
        fig, axes = plt.subplots(2, 3, figsize=(PLOT_WIDTH, PLOT_HEIGHT*1.5))
        axes = axes.flatten()
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
        
        for i, (joint, angles) in enumerate(rd["joint_angles"].items()):
            if i >= len(axes) or not angles:
                continue
                
            smoothed = smooth_data(angles)
            axes[i].plot(smoothed, color=colors[i % len(colors)], linewidth=2, label=joint.replace('_', ' ').title())
            
            # Mark gait phases
            for phase in gait_phases:
                phase_name, start, end = phase
                if start < len(smoothed) and end <= len(smoothed):
                    color = 'green' if phase_name == 'stance' else 'blue'
                    axes[i].axvspan(start, end, color=color, alpha=0.1)
            
            axes[i].set_title(f"{joint.replace('_', ' ').title()} Angle", fontsize=12)
            axes[i].set_xlabel("Frame")
            axes[i].set_ylabel("Angle (°)")
            axes[i].grid(True, alpha=0.3)
            axes[i].legend()
        
        # Hide unused subplots
        for i in range(len(rd["joint_angles"]), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # Foot trajectory analysis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(PLOT_WIDTH, PLOT_HEIGHT//1.5))
    
    # X trajectory (forward movement)
    ax1.plot(left_foot_x, 'r-', linewidth=2, label='Left Foot')
    ax1.plot(right_foot_x, 'b-', linewidth=2, label='Right Foot')
    ax1.set_title('Foot Movement (Forward/Backward)')
    ax1.set_xlabel('Frame')
    ax1.set_ylabel('Normalized X Position')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Y trajectory (vertical movement)
    ax2.plot(left_foot_y, 'r-', linewidth=2, label='Left Foot')
    ax2.plot(right_foot_y, 'b-', linewidth=2, label='Right Foot')
    ax2.set_title('Foot Movement (Vertical)')
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Normalized Y Position')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Output file info
    st.markdown("---")
    st.subheader("📁 Output Files")
    col1, col2 = st.columns(2)
    with col1:
        st.success(f"📄 **CSV Data:** `{rd['csv_path']}`")
    with col2:
        st.success(f"🎥 **Video:** `{rd['video_path']}`")
    
    # Analysis insights
    st.subheader("🔍 Clinical Insights")
    insights = []
    
    if cadence > 0:
        if cadence < 100:
            insights.append(f"🐌 **Slow walking pace**: {cadence:.0f} steps/min (normal: 100-120)")
        elif cadence > 120:
            insights.append(f"🏃 **Fast walking pace**: {cadence:.0f} steps/min (normal: 100-120)")
        else:
            insights.append(f"✅ **Normal walking pace**: {cadence:.0f} steps/min")
    
    step_symmetry = min(left_mean_step_length, right_mean_step_length) / max(left_mean_step_length, right_mean_step_length) if max(left_mean_step_length, right_mean_step_length) > 0 else 1
    if step_symmetry < 0.9:
        insights.append(f"⚠️ **Asymmetric gait**: Step symmetry {step_symmetry:.2f} (ideal: >0.95)")
    else:
        insights.append(f"✅ **Symmetric gait**: Good left-right balance")
    
    if rd["joint_angles"] and "left_knee" in rd["joint_angles"]:
        knee_rom = max(rd["joint_angles"]["left_knee"]) - min(rd["joint_angles"]["left_knee"])
        if knee_rom < 30:
            insights.append(f"⚠️ **Limited knee flexion**: {knee_rom:.1f}° ROM (normal: 40-60°)")
        elif knee_rom > 70:
            insights.append(f"📈 **High knee flexion**: {knee_rom:.1f}° ROM")
        else:
            insights.append(f"✅ **Normal knee ROM**: {knee_rom:.1f}°")
    
    for insight in insights:
        st.markdown(insight)
    
    if not insights:
        st.markdown("✅ **Overall**: Gait patterns appear within normal ranges")