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
SMOOTH_WINDOW = 5  # for smoothing angle data

def calculate_angle(a, b, c):
    """Calculate 3D angle between three points with 0° as straight leg"""
    a = np.array([a.x, a.y, a.z])
    b = np.array([b.x, b.y, b.z])
    c = np.array([c.x, c.y, c.z])
    
    # Vectors from joint to adjacent points
    ba = a - b
    bc = c - b
    
    # Calculate angle (0-180°)
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = degrees(acos(np.clip(cosine_angle, -1.0, 1.0)))
    
    return angle

def calculate_ankle_angle(heel, ankle, toe):
    """Calculate ankle angle with proper sign (negative for plantar flexion)"""
    heel = np.array([heel.x, heel.y, heel.z])
    ankle = np.array([ankle.x, ankle.y, ankle.z])
    toe = np.array([toe.x, toe.y, toe.z])
    
    # Vectors
    foot_vector = toe - heel
    shank_vector = ankle - heel
    
    # Cross product to determine direction
    cross = np.cross(shank_vector, foot_vector)
    sign = -1 if cross[2] < 0 else 1
    
    # Calculate angle (0° when foot is perpendicular to shank)
    cosine_angle = np.dot(shank_vector, foot_vector) / (np.linalg.norm(shank_vector) * np.linalg.norm(foot_vector))
    angle = sign * degrees(acos(np.clip(cosine_angle, -1.0, 1.0)))
    
    return angle

def detect_gait_phases(foot_distances, prominence=0.05):
    """Detect gait phases (stance/swing) from foot distances"""
    peaks, _ = find_peaks(foot_distances, prominence=prominence)
    valleys, _ = find_peaks(-foot_distances, prominence=prominence)
    
    # Create gait phase segments
    phases = []
    for i in range(len(peaks)-1):
        start = peaks[i]
        mid = valleys[np.where((valleys > peaks[i]) & (valleys < peaks[i+1]))[0][0] if len(valleys) > 0 else (start + peaks[i+1])//2
        end = peaks[i+1]
        phases.append(('stance', start, mid))
        phases.append(('swing', mid, end))
    
    return phases

def run_live_gait_analysis():
    st.title("🎥 Live Gait Analysis")
    st.markdown("""
    **Instructions:**
    1. Stand straight in the camera view (knees fully extended, ankles neutral)
    2. Press **Start Recording** and walk naturally
    3. Press **Stop Recording** when finished
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
            "video_path": None,
            "fps": 30,
            "video_writer": None,
            "initial_angles": None
        }

    # Camera controls
    col1, col2 = st.columns(2)
    with col1:
        zoom = st.slider("Zoom level", 1.0, 2.0, 1.0, 0.1)
    with col2:
        mirror = st.checkbox("Mirror view", True)

    # Recording controls
    if not st.session_state.recording:
        if st.button("▶️ Start Recording", type="primary"):
            st.session_state.recording = True
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs("outputs", exist_ok=True)
            video_file = f"outputs/gait_live_{timestamp}.mp4"
            
            # Initialize video writer
            cap = cv2.VideoCapture(0)
            width, height = int(cap.get(3)), int(cap.get(4))
            cap.release()
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            st.session_state.recorded_data["video_writer"] = cv2.VideoWriter(
                video_file, fourcc, 30.0, (width, height))
            st.session_state.recorded_data["video_path"] = video_file
            
            st.session_state.recorded_data.update({
                "left_foot": {"x": [], "y": [], "z": []},
                "right_foot": {"x": [], "y": [], "z": []},
                "joint_positions": defaultdict(list),
                "joint_angles": defaultdict(list),
                "frame_count": 0,
                "csv_path": f"outputs/gait_live_{timestamp}.csv",
                "fps": 30,
                "initial_angles": None
            })
    else:
        if st.button("⏹️ Stop Recording", type="primary"):
            st.session_state.recording = False
            if st.session_state.recorded_data["video_writer"] is not None:
                st.session_state.recorded_data["video_writer"].release()

    # Initialize MediaPipe
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    # Camera setup
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("❌ Cannot open webcam. Make sure it is connected and accessible.")
        return

    width, height = int(cap.get(3)), int(cap.get(4))
    frame_display = st.empty()

    # Main processing loop
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
        
        # Show frame
        frame_display.image(frame, channels="BGR", use_container_width=True)

        # Recording logic
        if st.session_state.recording:
            # Write frame to video
            if st.session_state.recorded_data["video_writer"] is not None:
                st.session_state.recorded_data["video_writer"].write(frame)
            
            if result.pose_landmarks:
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
                # Knee angles (0° when straight)
                left_knee_angle = calculate_angle(
                    lm[mp_pose.PoseLandmark.LEFT_HIP],
                    lm[mp_pose.PoseLandmark.LEFT_KNEE],
                    lm[mp_pose.PoseLandmark.LEFT_ANKLE]
                )
                right_knee_angle = calculate_angle(
                    lm[mp_pose.PoseLandmark.RIGHT_HIP],
                    lm[mp_pose.PoseLandmark.RIGHT_KNEE],
                    lm[mp_pose.PoseLandmark.RIGHT_ANKLE]
                )
                
                # Ankle angles (0° when neutral, negative for plantar flexion)
                left_ankle_angle = calculate_ankle_angle(
                    lm[mp_pose.PoseLandmark.LEFT_HEEL],
                    lm[mp_pose.PoseLandmark.LEFT_ANKLE],
                    lm[mp_pose.PoseLandmark.LEFT_FOOT_INDEX]
                )
                right_ankle_angle = calculate_ankle_angle(
                    lm[mp_pose.PoseLandmark.RIGHT_HEEL],
                    lm[mp_pose.PoseLandmark.RIGHT_ANKLE],
                    lm[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]
                )
                
                # Hip angles
                left_hip_angle = calculate_angle(
                    lm[mp_pose.PoseLandmark.LEFT_SHOULDER],
                    lm[mp_pose.PoseLandmark.LEFT_HIP],
                    lm[mp_pose.PoseLandmark.LEFT_KNEE]
                )
                right_hip_angle = calculate_angle(
                    lm[mp_pose.PoseLandmark.RIGHT_SHOULDER],
                    lm[mp_pose.PoseLandmark.RIGHT_HIP],
                    lm[mp_pose.PoseLandmark.RIGHT_KNEE]
                )
                
                # Store angles
                rd["joint_angles"]["left_knee"].append(left_knee_angle)
                rd["joint_angles"]["right_knee"].append(right_knee_angle)
                rd["joint_angles"]["left_ankle"].append(left_ankle_angle)
                rd["joint_angles"]["right_ankle"].append(right_ankle_angle)
                rd["joint_angles"]["left_hip"].append(left_hip_angle)
                rd["joint_angles"]["right_hip"].append(right_hip_angle)

                rd["frame_count"] += 1

        # Break loop when recording is stopped and we have some frames
        if not st.session_state.recording and st.session_state.recorded_data["frame_count"] > 0:
            break

    # Release resources
    cap.release()
    pose.close()

    # Save data and show results if recording was done
    if not st.session_state.recording and st.session_state.recorded_data["frame_count"] > 0:
        save_recording_data()
        display_gait_analysis_results()

def save_recording_data():
    """Save recorded data to files"""
    rd = st.session_state.recorded_data
    
    # Save CSV data
    with open(rd["csv_path"], 'w', newline='') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(['frame', 'left_foot_x', 'left_foot_y', 'left_foot_z',
                        'right_foot_x', 'right_foot_y', 'right_foot_z'] +
                       [f"{joint}_{coord}" for joint in rd["joint_positions"].keys() 
                        for coord in ['x', 'y', 'z']] +
                       list(rd["joint_angles"].keys()))
        
        # Write data row by row
        for i in range(rd["frame_count"]):
            row = [i]
            row.extend([rd["left_foot"]["x"][i], rd["left_foot"]["y"][i], rd["left_foot"]["z"][i]])
            row.extend([rd["right_foot"]["x"][i], rd["right_foot"]["y"][i], rd["right_foot"]["z"][i]])
            
            for joint in rd["joint_positions"].values():
                row.extend(joint[i])
                
            for angle in rd["joint_angles"].values():
                row.append(angle[i])
                
            writer.writerow(row)

def display_gait_analysis_results():
    """Display all gait analysis results after recording"""
    rd = st.session_state.recorded_data
    
    st.success("✅ Recording complete. Gait analysis results:")
    st.markdown("---")
    
    # Basic metrics
    duration = rd["frame_count"] / rd["fps"]
    left_foot_x = np.array(rd["left_foot"]["x"])
    left_foot_y = np.array(rd["left_foot"]["y"])
    right_foot_x = np.array(rd["right_foot"]["x"])
    right_foot_y = np.array(rd["right_foot"]["y"])
    
    # Calculate step distances and detect gait phases
    foot_dists = np.sqrt((left_foot_x - right_foot_x)**2 + (left_foot_y - right_foot_y)**2)
    gait_phases = detect_gait_phases(foot_dists)
    peaks, _ = find_peaks(foot_dists, prominence=0.05)
    num_steps = len(peaks)
    
    # Calculate mean angles for each gait phase
    phase_metrics = defaultdict(dict)
    for phase in gait_phases:
        phase_name, start, end = phase
        for joint, angles in rd["joint_angles"].items():
            phase_angles = angles[start:end]
            if phase_angles:
                phase_metrics[phase_name][f"{joint}_mean"] = np.mean(phase_angles)
                phase_metrics[phase_name][f"{joint}_min"] = min(phase_angles)
                phase_metrics[phase_name][f"{joint}_max"] = max(phase_angles)
    
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
    
    # Joint angles by phase
    st.subheader("🦵 Joint Angles by Gait Phase")
    
    # Create phase comparison data
    phase_comparison = []
    for joint in ['left_knee', 'right_knee', 'left_ankle', 'right_ankle', 'left_hip', 'right_hip']:
        stance_mean = phase_metrics['stance'].get(f"{joint}_mean", 0)
        swing_mean = phase_metrics['swing'].get(f"{joint}_mean", 0)
        
        phase_comparison.append({
            "Joint": joint.replace('_', ' ').title(),
            "Stance Mean (°)": f"{stance_mean:.1f}",
            "Swing Mean (°)": f"{swing_mean:.1f}",
            "Stance Min (°)": f"{phase_metrics['stance'].get(f'{joint}_min', 0):.1f}",
            "Stance Max (°)": f"{phase_metrics['stance'].get(f'{joint}_max', 0):.1f}",
            "Swing Min (°)": f"{phase_metrics['swing'].get(f'{joint}_min', 0):.1f}",
            "Swing Max (°)": f"{phase_metrics['swing'].get(f'{joint}_max', 0):.1f}"
        })
    
    st.table(pd.DataFrame(phase_comparison))
    
    # Visualization section
    st.subheader("📈 Gait Visualizations")
    
    # Foot distance plot with gait phases
    fig, ax = plt.subplots(figsize=(PLOT_WIDTH, PLOT_HEIGHT))
    ax.plot(foot_dists, label='Foot Distance')
    ax.plot(peaks, foot_dists[peaks], "rx", label='Detected Steps')
    
    # Color gait phases (green for stance, blue for swing)
    for phase in gait_phases:
        phase_name, start, end = phase
        color = 'lightgreen' if phase_name == 'stance' else 'lightblue'
        label = 'Stance Phase' if phase_name == 'stance' and start == 0 else 'Swing Phase' if start == 0 else ""
        ax.axvspan(start, end, color=color, alpha=0.3, label=label)
    
    ax.set_title("Step Detection with Gait Phases")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Normalized Distance")
    ax.legend()
    st.pyplot(fig)
    
    # Joint angles plots with gait phases
    fig, axes = plt.subplots(3, 2, figsize=(PLOT_WIDTH, PLOT_HEIGHT*3))
    axes = axes.flatten()
    
    for i, (joint, angles) in enumerate(rd["joint_angles"].items()):
        if i >= len(axes):
            break
        
        # Smooth angles for better visualization
        smoothed = smooth_data(angles)
        axes[i].plot(smoothed, label='Joint Angle')
        
        # Mark gait phases
        for phase in gait_phases:
            phase_name, start, end = phase
            color = 'green' if phase_name == 'stance' else 'blue'
            axes[i].axvspan(start, end, color=color, alpha=0.1)
        
        # Add horizontal line at 0° for reference
        axes[i].axhline(0, color='gray', linestyle='--', alpha=0.5)
        
        # Special formatting for ankle angles
        if 'ankle' in joint:
            axes[i].axhline(0, color='red', linestyle='-', alpha=0.3, label='Neutral Position')
            axes[i].set_ylabel("Angle (°)\n(Negative = Plantar Flexion)")
        else:
            axes[i].set_ylabel("Angle (°)")
        
        axes[i].set_title(f"{joint.replace('_', ' ').title()} Angle")
        axes[i].set_xlabel("Frame")
        axes[i].legend()
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Output file info
    st.markdown("---")
    st.subheader("📁 Output Files")
    st.text(f"CSV data file: {rd['csv_path']}")
    st.text(f"Video file: {rd['video_path']}")