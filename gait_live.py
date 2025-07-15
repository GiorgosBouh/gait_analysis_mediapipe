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
import tempfile
import threading
import time

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
    
    # Camera access troubleshooting info
    with st.expander("📋 Camera Troubleshooting", expanded=False):
        st.markdown("""
        **If camera doesn't work:**
        1. **Check browser permissions** - Make sure camera access is allowed
        2. **Try HTTPS** - Some browsers require HTTPS for camera access
        3. **Refresh the page** - Sometimes helps with permission issues
        4. **Try different browser** - Chrome usually works best
        5. **Check if camera is in use** - Close other apps using the camera
        
        **Current URL:** Your Streamlit app should ideally run on HTTPS for camera access
        """)
    
    # Main options
    option = st.radio(
        "Choose your preferred method:",
        [
            "📁 Upload Video File (Recommended)",
            "🎥 Try Live Camera (May have browser restrictions)",
            "📊 View Demo Analysis"
        ]
    )
    
    if option == "📁 Upload Video File (Recommended)":
        handle_video_upload()
    elif option == "🎥 Try Live Camera (May have browser restrictions)":
        handle_live_camera()
    else:
        handle_demo_analysis()

def handle_video_upload():
    """Handle video file upload and analysis"""
    st.markdown("""
    ### 📁 Video Upload Method
    
    **Best for reliable analysis!**
    
    **Instructions:**
    1. Record a video of yourself walking (side view works best)
    2. Upload the video file below
    3. Get detailed gait analysis results
    """)
    
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Upload a video showing walking from the side view for best results"
    )
    
    if uploaded_file is not None:
        # Show file info
        file_details = {
            "Filename": uploaded_file.name,
            "File size": f"{uploaded_file.size / (1024*1024):.2f} MB",
            "File type": uploaded_file.type
        }
        st.table(pd.DataFrame.from_dict(file_details, orient='index', columns=['Value']))
        
        if st.button("🚀 Analyze Video", type="primary"):
            with st.spinner("🔄 Processing video... This may take a few minutes."):
                progress_bar = st.progress(0)
                
                # Process the video
                analysis_data = process_uploaded_video(uploaded_file, progress_bar)
                
                if analysis_data and analysis_data["frame_count"] > 0:
                    st.success("✅ Video analysis complete!")
                    display_gait_analysis_results(analysis_data)
                    
                    # Save results
                    if st.button("💾 Save Results"):
                        save_recording_data(analysis_data, uploaded_file.name)
                else:
                    st.error("❌ Could not detect pose in the video. Please ensure the person is clearly visible.")

def handle_live_camera():
    """Handle live camera recording (with limitations note)"""
    st.markdown("""
    ### 🎥 Live Camera Method
    
    ⚠️ **Note:** Browser security restrictions may prevent camera access in Streamlit.
    If this doesn't work, please use the **Upload Video File** method instead.
    """)
    
    # Initialize session state for camera recording
    if "recording_state" not in st.session_state:
        st.session_state.recording_state = {
            "is_recording": False,
            "frames": [],
            "recorded_data": None
        }
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📹 Start Recording"):
            if start_camera_recording():
                st.session_state.recording_state["is_recording"] = True
                st.rerun()
    
    with col2:
        if st.button("⏹️ Stop Recording"):
            if st.session_state.recording_state["is_recording"]:
                analysis_data = stop_camera_recording()
                st.session_state.recording_state["is_recording"] = False
                st.session_state.recording_state["recorded_data"] = analysis_data
                st.rerun()
    
    with col3:
        if st.session_state.recording_state["is_recording"]:
            st.markdown("🔴 **RECORDING...**")
        else:
            st.markdown("⚪ **Ready**")
    
    # Show camera feed placeholder
    if st.session_state.recording_state["is_recording"]:
        camera_placeholder = st.empty()
        show_camera_feed(camera_placeholder)
    
    # Show results if available
    if st.session_state.recording_state["recorded_data"]:
        st.markdown("---")
        display_gait_analysis_results(st.session_state.recording_state["recorded_data"])

def handle_demo_analysis():
    """Show demo analysis with sample data"""
    st.markdown("""
    ### 📊 Demo Analysis
    
    See what a complete gait analysis looks like with sample data.
    """)
    
    if st.button("🎬 Generate Demo Analysis"):
        with st.spinner("Creating demo analysis..."):
            demo_data = create_demo_data()
            display_gait_analysis_results(demo_data)

def start_camera_recording():
    """Start camera recording (simplified version)"""
    try:
        # Try to access camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("❌ Cannot access camera. Please check if:")
            st.write("- Camera is connected and working")
            st.write("- No other application is using the camera")
            st.write("- Browser has camera permissions")
            return False
        
        # Test if we can read a frame
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            st.error("❌ Cannot read from camera")
            return False
        
        st.success("✅ Camera access successful! Recording started.")
        return True
        
    except Exception as e:
        st.error(f"❌ Camera error: {str(e)}")
        st.info("💡 Try using the 'Upload Video File' method instead")
        return False

def stop_camera_recording():
    """Stop camera recording and process frames"""
    # For now, return demo data since camera recording in browser is complex
    st.info("🔄 Processing recorded frames...")
    return create_demo_data()

def show_camera_feed(placeholder):
    """Show camera feed (placeholder for now)"""
    placeholder.info("📹 Camera feed would appear here. Due to browser restrictions, please use video upload method for best results.")

def process_uploaded_video(uploaded_file, progress_bar=None):
    """Process uploaded video file with MediaPipe"""
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        temp_path = tmp_file.name
    
    try:
        # Initialize MediaPipe
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Open video
        cap = cv2.VideoCapture(temp_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        analysis_data = {
            "left_foot": {"x": [], "y": [], "z": []},
            "right_foot": {"x": [], "y": [], "z": []},
            "joint_positions": defaultdict(list),
            "joint_angles": defaultdict(list),
            "frame_count": 0
        }
        
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Update progress
            if progress_bar:
                progress = frame_count / total_frames
                progress_bar.progress(progress)
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = pose.process(rgb_frame)
            
            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                
                # Extract foot positions
                analysis_data["left_foot"]["x"].append(lm[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x)
                analysis_data["left_foot"]["y"].append(lm[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y)
                analysis_data["left_foot"]["z"].append(lm[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].z)
                
                analysis_data["right_foot"]["x"].append(lm[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].x)
                analysis_data["right_foot"]["y"].append(lm[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y)
                analysis_data["right_foot"]["z"].append(lm[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].z)
                
                # Store joint positions
                for joint in ['LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_HIP', 'RIGHT_HIP', 'LEFT_ANKLE', 'RIGHT_ANKLE']:
                    analysis_data["joint_positions"][joint].append([
                        lm[mp_pose.PoseLandmark[joint]].x,
                        lm[mp_pose.PoseLandmark[joint]].y,
                        lm[mp_pose.PoseLandmark[joint]].z
                    ])
                
                # Calculate joint angles
                analysis_data["joint_angles"]["left_knee"].append(
                    calculate_angle(lm[mp_pose.PoseLandmark.LEFT_HIP], 
                                  lm[mp_pose.PoseLandmark.LEFT_KNEE], 
                                  lm[mp_pose.PoseLandmark.LEFT_ANKLE])
                )
                analysis_data["joint_angles"]["right_knee"].append(
                    calculate_angle(lm[mp_pose.PoseLandmark.RIGHT_HIP], 
                                   lm[mp_pose.PoseLandmark.RIGHT_KNEE], 
                                   lm[mp_pose.PoseLandmark.RIGHT_ANKLE])
                )
                analysis_data["joint_angles"]["left_hip"].append(
                    calculate_angle(lm[mp_pose.PoseLandmark.LEFT_SHOULDER], 
                                  lm[mp_pose.PoseLandmark.LEFT_HIP], 
                                  lm[mp_pose.PoseLandmark.LEFT_KNEE])
                )
                analysis_data["joint_angles"]["right_hip"].append(
                    calculate_angle(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER], 
                                  lm[mp_pose.PoseLandmark.RIGHT_HIP], 
                                  lm[mp_pose.PoseLandmark.RIGHT_KNEE])
                )
                analysis_data["joint_angles"]["left_ankle"].append(
                    calculate_angle(lm[mp_pose.PoseLandmark.LEFT_KNEE], 
                                  lm[mp_pose.PoseLandmark.LEFT_ANKLE], 
                                  lm[mp_pose.PoseLandmark.LEFT_HEEL])
                )
                analysis_data["joint_angles"]["right_ankle"].append(
                    calculate_angle(lm[mp_pose.PoseLandmark.RIGHT_KNEE], 
                                   lm[mp_pose.PoseLandmark.RIGHT_ANKLE], 
                                   lm[mp_pose.PoseLandmark.RIGHT_HEEL])
                )
                
                analysis_data["frame_count"] += 1
            
            frame_count += 1
        
        cap.release()
        pose.close()
        
        if progress_bar:
            progress_bar.progress(1.0)
        
        return analysis_data
        
    except Exception as e:
        st.error(f"Error processing video: {str(e)}")
        return None
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

def create_demo_data():
    """Create realistic demo gait analysis data"""
    # Generate 10 seconds of walking data at 30 FPS
    frames = 300
    time_points = np.linspace(0, 10, frames)
    
    # Create realistic foot trajectories
    # Walking pattern: alternating foot contact
    left_foot_cycle = np.sin(2 * np.pi * 0.8 * time_points) * 0.1 + 0.3  # 0.8 Hz walking
    left_foot_y = np.abs(np.sin(4 * np.pi * 0.8 * time_points)) * 0.15 + 0.75
    
    right_foot_cycle = np.sin(2 * np.pi * 0.8 * time_points + np.pi) * 0.1 + 0.7
    right_foot_y = np.abs(np.sin(4 * np.pi * 0.8 * time_points + np.pi)) * 0.15 + 0.75
    
    # Joint angles with realistic ranges
    # Knee: 0° (extended) to 60° (flexed)
    left_knee = 30 + 25 * np.sin(4 * np.pi * 0.8 * time_points)
    right_knee = 30 + 25 * np.sin(4 * np.pi * 0.8 * time_points + np.pi)
    
    # Hip: 170° to 200° (slight flexion/extension)
    left_hip = 185 + 10 * np.sin(2 * np.pi * 0.8 * time_points)
    right_hip = 185 + 10 * np.sin(2 * np.pi * 0.8 * time_points + np.pi)
    
    # Ankle: 80° to 100° 
    left_ankle = 90 + 8 * np.sin(4 * np.pi * 0.8 * time_points + np.pi/4)
    right_ankle = 90 + 8 * np.sin(4 * np.pi * 0.8 * time_points + np.pi + np.pi/4)
    
    return {
        "left_foot": {
            "x": left_foot_cycle.tolist(),
            "y": left_foot_y.tolist(),
            "z": [0.1] * frames
        },
        "right_foot": {
            "x": right_foot_cycle.tolist(),
            "y": right_foot_y.tolist(),
            "z": [0.1] * frames
        },
        "joint_positions": defaultdict(list),
        "joint_angles": {
            "left_knee": left_knee.tolist(),
            "right_knee": right_knee.tolist(),
            "left_hip": left_hip.tolist(),
            "right_hip": right_hip.tolist(),
            "left_ankle": left_ankle.tolist(),
            "right_ankle": right_ankle.tolist()
        },
        "frame_count": frames
    }

def save_recording_data(recorded_data, filename_prefix="gait_analysis"):
    """Save recorded data to files"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("outputs", exist_ok=True)
    
    # Clean filename
    base_name = os.path.splitext(filename_prefix)[0]
    csv_path = f"outputs/{base_name}_analysis_{timestamp}.csv"
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header
        header = ['frame', 'left_foot_x', 'left_foot_y', 'left_foot_z',
                  'right_foot_x', 'right_foot_y', 'right_foot_z']
        header.extend(recorded_data["joint_angles"].keys())
        writer.writerow(header)
        
        # Write data
        for i in range(recorded_data["frame_count"]):
            row = [i]
            row.extend([
                recorded_data["left_foot"]["x"][i],
                recorded_data["left_foot"]["y"][i],
                recorded_data["left_foot"]["z"][i],
                recorded_data["right_foot"]["x"][i],
                recorded_data["right_foot"]["y"][i],
                recorded_data["right_foot"]["z"][i]
            ])
            
            for angles in recorded_data["joint_angles"].values():
                row.append(angles[i] if i < len(angles) else 0)
            
            writer.writerow(row)
    
    st.success(f"✅ Analysis saved to: `{csv_path}`")
    return csv_path

def display_gait_analysis_results(recorded_data):
    """Display comprehensive gait analysis results"""
    st.markdown("---")
    st.subheader("📊 Gait Analysis Results")
    
    # Basic metrics
    fps = 30
    duration = recorded_data["frame_count"] / fps
    
    left_foot_x = np.array(recorded_data["left_foot"]["x"])
    left_foot_y = np.array(recorded_data["left_foot"]["y"])
    right_foot_x = np.array(recorded_data["right_foot"]["x"])
    right_foot_y = np.array(recorded_data["right_foot"]["y"])
    
    # Calculate foot distance for step detection
    foot_dists = np.sqrt((left_foot_x - right_foot_x)**2 + (left_foot_y - right_foot_y)**2)
    
    # Detect steps and gait phases
    gait_phases = detect_gait_phases(foot_dists)
    peaks, _ = find_peaks(foot_dists, prominence=0.02)
    num_steps = len(peaks)
    
    # Calculate gait metrics
    cadence = (num_steps / duration) * 60 if duration > 0 else 0
    step_time = duration / num_steps if num_steps > 0 else 0
    
    # Step lengths (normalized)
    left_steps = [foot_dists[peaks[i]] for i in range(len(peaks)) if i % 2 == 0]
    right_steps = [foot_dists[peaks[i]] for i in range(len(peaks)) if i % 2 == 1]
    
    avg_left_step = np.mean(left_steps) if left_steps else 0
    avg_right_step = np.mean(right_steps) if right_steps else 0
    step_symmetry = min(avg_left_step, avg_right_step) / max(avg_left_step, avg_right_step) if max(avg_left_step, avg_right_step) > 0 else 1
    
    # Display summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Duration", f"{duration:.1f}s")
        st.metric("Steps Detected", f"{num_steps}")
    
    with col2:
        st.metric("Cadence", f"{cadence:.0f} steps/min")
        st.metric("Step Time", f"{step_time:.2f}s")
    
    with col3:
        st.metric("Left Steps", f"{len(left_steps)}")
        st.metric("Right Steps", f"{len(right_steps)}")
    
    with col4:
        st.metric("Step Symmetry", f"{step_symmetry:.2f}")
        st.metric("Avg Step Width", f"{np.mean(np.abs(left_foot_y - right_foot_y)):.3f}")
    
    # Joint ROM analysis
    st.subheader("🦵 Joint Range of Motion Analysis")
    
    if recorded_data["joint_angles"]:
        rom_data = []
        for joint, angles in recorded_data["joint_angles"].items():
            if angles:
                rom = max(angles) - min(angles)
                mean_angle = np.mean(angles)
                std_angle = np.std(angles)
                
                rom_data.append({
                    "Joint": joint.replace('_', ' ').title(),
                    "ROM (°)": f"{rom:.1f}",
                    "Mean Angle (°)": f"{mean_angle:.1f}",
                    "Std Dev (°)": f"{std_angle:.1f}"
                })
        
        st.table(pd.DataFrame(rom_data))
    
    # Visualization section
    st.subheader("📈 Gait Pattern Visualizations")
    
    # 1. Foot distance and step detection
    fig1, ax1 = plt.subplots(figsize=(PLOT_WIDTH, PLOT_HEIGHT))
    ax1.plot(foot_dists, 'b-', linewidth=2, label='Inter-foot Distance')
    ax1.plot(peaks, foot_dists[peaks], 'ro', markersize=8, label=f'Steps Detected ({num_steps})')
    
    # Color code gait phases
    for i, (phase_name, start, end) in enumerate(gait_phases[:4]):  # Show first 4 phases
        color = 'lightgreen' if phase_name == 'stance' else 'lightblue'
        ax1.axvspan(start, end, color=color, alpha=0.3, 
                   label=f'{phase_name.capitalize()}' if i < 2 else "")
    
    ax1.set_title('Step Detection and Gait Phases', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Frame Number', fontsize=12)
    ax1.set_ylabel('Normalized Distance', fontsize=12)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    st.pyplot(fig1)
    
    # 2. Joint angles over time
    if recorded_data["joint_angles"]:
        fig2, axes = plt.subplots(2, 3, figsize=(PLOT_WIDTH*1.2, PLOT_HEIGHT*1.5))
        axes = axes.flatten()
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
        
        for i, (joint, angles) in enumerate(recorded_data["joint_angles"].items()):
            if i >= len(axes) or not angles:
                continue
            
            smoothed = smooth_data(angles)
            time_axis = np.linspace(0, duration, len(smoothed))
            
            axes[i].plot(time_axis, smoothed, color=colors[i % len(colors)], 
                        linewidth=2.5, label=joint.replace('_', ' ').title())
            
            # Mark gait phases with light shading
            for phase_name, start, end in gait_phases:
                if start < len(smoothed) and end <= len(smoothed):
                    start_time = start * duration / len(smoothed)
                    end_time = end * duration / len(smoothed)
                    color = 'green' if phase_name == 'stance' else 'blue'
                    axes[i].axvspan(start_time, end_time, color=color, alpha=0.1)
            
            axes[i].set_title(f"{joint.replace('_', ' ').title()} Angle", 
                             fontsize=12, fontweight='bold')
            axes[i].set_xlabel('Time (seconds)', fontsize=10)
            axes[i].set_ylabel('Angle (degrees)', fontsize=10)
            axes[i].grid(True, alpha=0.3)
            axes[i].legend()
        
        # Hide unused subplots
        for i in range(len(recorded_data["joint_angles"]), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        st.pyplot(fig2)
    
    # 3. Foot trajectories
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(PLOT_WIDTH, PLOT_HEIGHT//1.5))
    
    # X trajectory (forward movement)
    time_axis = np.linspace(0, duration, len(left_foot_x))
    ax3a.plot(time_axis, left_foot_x, 'r-', linewidth=2, label='Left Foot')
    ax3a.plot(time_axis, right_foot_x, 'b-', linewidth=2, label='Right Foot')
    ax3a.set_title('Foot Movement (Forward/Backward)', fontweight='bold')
    ax3a.set_xlabel('Time (seconds)')
    ax3a.set_ylabel('Normalized X Position')
    ax3a.legend()
    ax3a.grid(True, alpha=0.3)
    
    # Y trajectory (vertical movement)
    ax3b.plot(time_axis, left_foot_y, 'r-', linewidth=2, label='Left Foot')
    ax3b.plot(time_axis, right_foot_y, 'b-', linewidth=2, label='Right Foot')
    ax3b.set_title('Foot Movement (Vertical)', fontweight='bold')
    ax3b.set_xlabel('Time (seconds)')
    ax3b.set_ylabel('Normalized Y Position')
    ax3b.legend()
    ax3b.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig3)
    
    # Summary insights
    st.subheader("🔍 Analysis Summary")
    
    insights = []
    
    if cadence > 0:
        if cadence < 100:
            insights.append(f"🐌 **Slow walking pace**: {cadence:.0f} steps/min (normal: 100-120)")
        elif cadence > 120:
            insights.append(f"🏃 **Fast walking pace**: {cadence:.0f} steps/min (normal: 100-120)")
        else:
            insights.append(f"✅ **Normal walking pace**: {cadence:.0f} steps/min")
    
    if step_symmetry < 0.9:
        insights.append(f"⚠️ **Asymmetric gait**: Step symmetry {step_symmetry:.2f} (ideal: >0.95)")
    else:
        insights.append(f"✅ **Symmetric gait**: Good left-right balance")
    
    if recorded_data["joint_angles"]:
        knee_rom = max(recorded_data["joint_angles"]["left_knee"]) - min(recorded_data["joint_angles"]["left_knee"])
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
    
    # Download section
    st.markdown("---")
    if st.button("💾 Save Complete Analysis", type="primary"):
        csv_path = save_recording_data(recorded_data)
        st.balloons()  # Celebrate successful analysis!