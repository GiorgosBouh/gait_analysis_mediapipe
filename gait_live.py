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

def smooth_data(data, window_size=SMOOTH_WINDOW):
    """Apply simple moving average smoothing"""
    return savgol_filter(data, window_size, 2) if len(data) > window_size else data

def calculate_knee_angle(a, b, c):
    """Calculate knee angle with 0° as straight leg (180° in conventional terms)"""
    a = np.array([a.x, a.y, a.z])
    b = np.array([b.x, b.y, b.z])
    c = np.array([c.x, c.y, c.z])
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = 180 - degrees(acos(np.clip(cosine_angle, -1.0, 1.0)))  # Convert to conventional measurement
    
    return angle

def calculate_ankle_angle(heel, ankle, toe):
    """Calculate ankle angle with proper sign (0° when foot is perpendicular to shank)"""
    heel = np.array([heel.x, heel.y, heel.z])
    ankle = np.array([ankle.x, ankle.y, ankle.z])
    toe = np.array([toe.x, toe.y, toe.z])
    
    # Vectors
    foot_vector = toe - heel
    shank_vector = ankle - heel
    
    # Calculate angle (90° when foot is perpendicular to shank)
    cosine_angle = np.dot(shank_vector, foot_vector) / (np.linalg.norm(shank_vector) * np.linalg.norm(foot_vector))
    angle = 90 - degrees(acos(np.clip(cosine_angle, -1.0, 1.0)))  # Convert to conventional measurement
    
    # Determine direction (negative for plantar flexion)
    cross = np.cross(shank_vector, foot_vector)
    if cross[2] < 0:
        angle = -angle
    
    return angle

def calculate_hip_angle(a, b, c):
    """Calculate hip angle in conventional terms (0° when standing straight)"""
    a = np.array([a.x, a.y, a.z])
    b = np.array([b.x, b.y, b.z])
    c = np.array([c.x, c.y, c.z])
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = degrees(acos(np.clip(cosine_angle, -1.0, 1.0)))
    
    return angle

def detect_gait_phases(foot_distances, prominence=0.05):
    """Detect gait phases (stance/swing) from foot distances"""
    peaks, _ = find_peaks(foot_distances, prominence=prominence)
    valleys, _ = find_peaks(-foot_distances, prominence=prominence)
    
    phases = []
    for i in range(len(peaks) - 1):
        start = peaks[i]
        end = peaks[i + 1]

        # Filter valleys between current and next peak
        between_valleys = valleys[(valleys > start) & (valleys < end)]

        if len(between_valleys) > 0:
            mid = between_valleys[0]
        else:
            mid = (start + end) // 2  # fallback midpoint

        phases.append(('stance', start, mid))
        phases.append(('swing', mid, end))

    return phases

def run_live_gait_analysis():
    # [Previous implementation remains the same until angle calculations]
    
    # In the recording logic section, replace the angle calculations with:
    if result.pose_landmarks:
        lm = result.pose_landmarks.landmark
        rd = st.session_state.recorded_data

        # Calculate and store joint angles using conventional measurements
        left_knee_angle = calculate_knee_angle(
            lm[mp_pose.PoseLandmark.LEFT_HIP],
            lm[mp_pose.PoseLandmark.LEFT_KNEE],
            lm[mp_pose.PoseLandmark.LEFT_ANKLE]
        )
        right_knee_angle = calculate_knee_angle(
            lm[mp_pose.PoseLandmark.RIGHT_HIP],
            lm[mp_pose.PoseLandmark.RIGHT_KNEE],
            lm[mp_pose.PoseLandmark.RIGHT_ANKLE]
        )
        
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
        
        left_hip_angle = calculate_hip_angle(
            lm[mp_pose.PoseLandmark.LEFT_SHOULDER],
            lm[mp_pose.PoseLandmark.LEFT_HIP],
            lm[mp_pose.PoseLandmark.LEFT_KNEE]
        )
        right_hip_angle = calculate_hip_angle(
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

    # [Rest of the implementation remains the same]

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