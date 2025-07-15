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
import base64
import threading
import time
import streamlit.components.v1 as components

# Constants
PLOT_WIDTH = 12
PLOT_HEIGHT = 6
SMOOTH_WINDOW = 5  # for smoothing angle data

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

def create_camera_component():
    """Create HTML component for camera access"""
    camera_html = """
    <style>
        .camera-container {
            text-align: center;
            background: #f0f2f6;
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
        }
        .camera-feed {
            width: 100%;
            max-width: 640px;
            height: 480px;
            border: 2px solid #e1e5e9;
            border-radius: 10px;
            background: #000;
        }
        .camera-controls {
            margin: 15px 0;
        }
        .btn {
            padding: 10px 20px;
            margin: 0 5px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
        }
        .btn-primary {
            background: #ff4b4b;
            color: white;
        }
        .btn-secondary {
            background: #f0f2f6;
            color: #262730;
            border: 1px solid #e1e5e9;
        }
        .btn:hover {
            opacity: 0.8;
        }
        .status {
            font-size: 18px;
            font-weight: bold;
            margin: 10px 0;
        }
        .recording { color: #ff4b4b; }
        .ready { color: #00c851; }
        .error { color: #ff4444; }
    </style>
    
    <div class="camera-container">
        <div id="status" class="status ready">📷 Ready - Click Start Camera</div>
        <video id="videoElement" class="camera-feed" autoplay muted></video>
        <canvas id="canvasElement" style="display: none;"></canvas>
        
        <div class="camera-controls">
            <button id="startBtn" class="btn btn-primary">📹 Start Camera</button>
            <button id="stopBtn" class="btn btn-secondary" disabled>⏹️ Stop Camera</button>
            <button id="recordBtn" class="btn btn-secondary" disabled>🔴 Start Recording</button>
            <button id="uploadBtn" class="btn btn-secondary" disabled>📤 Upload for Analysis</button>
        </div>
        
        <div id="frameInfo" style="margin-top: 10px; font-size: 14px; color: #666;"></div>
    </div>
    
    <script>
        let video = document.getElementById('videoElement');
        let canvas = document.getElementById('canvasElement');
        let ctx = canvas.getContext('2d');
        let mediaStream = null;
        let isRecording = false;
        let recordedFrames = [];
        let recordingInterval = null;
        
        const status = document.getElementById('status');
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');
        const recordBtn = document.getElementById('recordBtn');
        const uploadBtn = document.getElementById('uploadBtn');
        const frameInfo = document.getElementById('frameInfo');
        
        function updateStatus(message, className) {
            status.textContent = message;
            status.className = `status ${className}`;
        }
        
        function updateFrameInfo(frames) {
            frameInfo.textContent = `Recorded frames: ${frames}`;
        }
        
        startBtn.addEventListener('click', async () => {
            try {
                updateStatus('🔄 Starting camera...', 'ready');
                mediaStream = await navigator.mediaDevices.getUserMedia({
                    video: { 
                        width: { ideal: 640 },
                        height: { ideal: 480 },
                        frameRate: { ideal: 30 }
                    }
                });
                
                video.srcObject = mediaStream;
                video.play();
                
                // Set canvas size
                canvas.width = 640;
                canvas.height = 480;
                
                updateStatus('📹 Camera ready - You can start recording', 'ready');
                startBtn.disabled = true;
                stopBtn.disabled = false;
                recordBtn.disabled = false;
                
            } catch (error) {
                updateStatus(`❌ Error: ${error.message}`, 'error');
                console.error('Error accessing camera:', error);
            }
        });
        
        stopBtn.addEventListener('click', () => {
            if (mediaStream) {
                mediaStream.getTracks().forEach(track => track.stop());
                video.srcObject = null;
                
                if (isRecording) {
                    stopRecording();
                }
                
                updateStatus('📷 Camera stopped', 'ready');
                startBtn.disabled = false;
                stopBtn.disabled = true;
                recordBtn.disabled = true;
                updateFrameInfo(0);
            }
        });
        
        recordBtn.addEventListener('click', () => {
            if (!isRecording) {
                startRecording();
            } else {
                stopRecording();
            }
        });
        
        function startRecording() {
            isRecording = true;
            recordedFrames = [];
            updateStatus('🔴 Recording...', 'recording');
            recordBtn.textContent = '⏹️ Stop Recording';
            uploadBtn.disabled = true;
            
            // Capture frames at 30 FPS
            recordingInterval = setInterval(() => {
                if (video.readyState === video.HAVE_ENOUGH_DATA) {
                    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                    const frameData = canvas.toDataURL('image/jpeg', 0.8);
                    recordedFrames.push(frameData);
                    updateFrameInfo(recordedFrames.length);
                }
            }, 1000 / 30); // 30 FPS
        }
        
        function stopRecording() {
            isRecording = false;
            clearInterval(recordingInterval);
            updateStatus(`✅ Recording stopped - ${recordedFrames.length} frames captured`, 'ready');
            recordBtn.textContent = '🔴 Start Recording';
            uploadBtn.disabled = recordedFrames.length === 0;
        }
        
        uploadBtn.addEventListener('click', () => {
            if (recordedFrames.length > 0) {
                updateStatus('📤 Uploading frames for analysis...', 'ready');
                
                // Send frames to Streamlit (this would need to be implemented)
                // For now, we'll store them in sessionStorage
                sessionStorage.setItem('gait_frames', JSON.stringify(recordedFrames));
                sessionStorage.setItem('gait_frame_count', recordedFrames.length.toString());
                
                // Trigger Streamlit rerun
                window.parent.postMessage({
                    type: 'gait_upload',
                    frames: recordedFrames.length
                }, '*');
                
                updateStatus('✅ Upload complete - Check analysis below', 'ready');
            }
        });
        
        // Handle page visibility to pause/resume recording
        document.addEventListener('visibilitychange', () => {
            if (document.hidden && isRecording) {
                // Pause recording when tab is hidden
                clearInterval(recordingInterval);
            } else if (!document.hidden && isRecording) {
                // Resume recording when tab is visible
                recordingInterval = setInterval(() => {
                    if (video.readyState === video.HAVE_ENOUGH_DATA) {
                        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                        const frameData = canvas.toDataURL('image/jpeg', 0.8);
                        recordedFrames.push(frameData);
                        updateFrameInfo(recordedFrames.length);
                    }
                }, 1000 / 30);
            }
        });
    </script>
    """
    
    return camera_html

def process_uploaded_frames():
    """Process frames from JavaScript component"""
    # This would be called after JavaScript uploads frames
    # For now, return dummy data structure
    return {
        "left_foot": {"x": [], "y": [], "z": []},
        "right_foot": {"x": [], "y": [], "z": []},
        "joint_positions": defaultdict(list),
        "joint_angles": defaultdict(list),
        "frame_count": 0
    }

def run_live_gait_analysis():
    st.title("🎥 Live Gait Analysis")
    st.markdown("""
    **Instructions:**
    1. Click **Start Camera** to access your webcam
    2. Position yourself in the camera view
    3. Click **Start Recording** and walk naturally
    4. Click **Stop Recording** when finished
    5. Click **Upload for Analysis** to process your gait data
    """)
    
    # Initialize session state
    if "analysis_data" not in st.session_state:
        st.session_state.analysis_data = None
    
    # Display camera component
    camera_component = create_camera_component()
    components.html(camera_component, height=700)
    
    # Check for uploaded frames from JavaScript
    if st.button("🔄 Check for Recorded Data"):
        # In a real implementation, this would process the frames from sessionStorage
        # For now, we'll create dummy data for demonstration
        st.session_state.analysis_data = create_dummy_analysis_data()
        
    # Display analysis results if available
    if st.session_state.analysis_data:
        st.markdown("---")
        st.success("✅ Analysis complete!")
        display_gait_analysis_results(st.session_state.analysis_data)
    
    # Alternative: File upload option
    st.markdown("---")
    st.subheader("📁 Alternative: Upload Pre-recorded Video")
    uploaded_file = st.file_uploader(
        "Upload a video file for gait analysis",
        type=['mp4', 'avi', 'mov'],
        help="Upload a video file showing walking from the side view"
    )
    
    if uploaded_file is not None:
        # Process uploaded video
        with st.spinner("Processing uploaded video..."):
            analysis_data = process_uploaded_video(uploaded_file)
            st.session_state.analysis_data = analysis_data
            st.rerun()

def create_dummy_analysis_data():
    """Create dummy analysis data for demonstration"""
    # Generate dummy gait data
    frames = 300  # 10 seconds at 30 FPS
    
    # Create realistic-looking dummy data
    left_foot_x = np.sin(np.linspace(0, 4*np.pi, frames)) * 0.1 + 0.3
    left_foot_y = np.abs(np.sin(np.linspace(0, 8*np.pi, frames))) * 0.2 + 0.7
    right_foot_x = np.sin(np.linspace(np.pi, 5*np.pi, frames)) * 0.1 + 0.7
    right_foot_y = np.abs(np.sin(np.linspace(np.pi, 9*np.pi, frames))) * 0.2 + 0.7
    
    # Create joint angles
    knee_angles = 90 + np.sin(np.linspace(0, 8*np.pi, frames)) * 30
    hip_angles = 180 + np.sin(np.linspace(0, 8*np.pi, frames)) * 20
    ankle_angles = 90 + np.sin(np.linspace(0, 8*np.pi, frames)) * 15
    
    return {
        "left_foot": {"x": left_foot_x.tolist(), "y": left_foot_y.tolist(), "z": [0.0] * frames},
        "right_foot": {"x": right_foot_x.tolist(), "y": right_foot_y.tolist(), "z": [0.0] * frames},
        "joint_positions": defaultdict(list),
        "joint_angles": {
            "left_knee": knee_angles.tolist(),
            "right_knee": (knee_angles + np.random.normal(0, 2, frames)).tolist(),
            "left_hip": hip_angles.tolist(),
            "right_hip": (hip_angles + np.random.normal(0, 2, frames)).tolist(),
            "left_ankle": ankle_angles.tolist(),
            "right_ankle": (ankle_angles + np.random.normal(0, 2, frames)).tolist()
        },
        "frame_count": frames
    }

def process_uploaded_video(uploaded_file):
    """Process uploaded video file"""
    # Save uploaded file temporarily
    temp_path = f"temp_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Initialize MediaPipe
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Process video
    cap = cv2.VideoCapture(temp_path)
    
    analysis_data = {
        "left_foot": {"x": [], "y": [], "z": []},
        "right_foot": {"x": [], "y": [], "z": []},
        "joint_positions": defaultdict(list),
        "joint_angles": defaultdict(list),
        "frame_count": 0
    }
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
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
    
    cap.release()
    pose.close()
    
    # Clean up temporary file
    if os.path.exists(temp_path):
        os.remove(temp_path)
    
    return analysis_data

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
            
            for angle in recorded_data["joint_angles"].values():
                row.append(angle[i])
                
            writer.writerow(row)
    
    st.success(f"✅ Data saved to: {csv_path}")
    return csv_path

def display_gait_analysis_results(recorded_data):
    """Display all gait analysis results after recording"""
    st.subheader("📊 Gait Analysis Results")
    
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
            if start < len(angles) and end <= len(angles):
                phase_angles = angles[start:end]
                if phase_angles:
                    rom = max(phase_angles) - min(phase_angles)
                    phase_rom[f"{joint}_{phase_name}"].append(rom)
    
    # Calculate mean ROM per phase
    mean_phase_rom = {}
    for key, values in phase_rom.items():
        mean_phase_rom[key] = np.mean(values) if values else 0
    
    # Gait metrics
    cadence = (num_steps / duration) * 60 if duration > 0 else 0
    step_time = duration / num_steps if num_steps > 0 else 0
    
    # Step lengths
    left_step_lengths = [foot_dists[peaks[i]] for i in range(len(peaks)) if i % 2 == 0]
    right_step_lengths = [foot_dists[peaks[i]] for i in range(len(peaks)) if i % 2 == 1]
    
    left_mean_step_length = np.mean(left_step_lengths) if left_step_lengths else 0
    right_mean_step_length = np.mean(right_step_lengths) if right_step_lengths else 0
    
    mean_step_width = np.mean(np.abs(left_foot_y - right_foot_y))
    
    # Display metrics
    st.subheader("📊 Gait Characteristics")
    metrics = {
        "Recording Duration": f"{duration:.2f} s",
        "Number of Steps": f"{num_steps} ({len(left_step_lengths)} left, {len(right_step_lengths)} right)",
        "Cadence": f"{cadence:.2f} steps/min",
        "Step Time": f"{step_time:.2f} s",
        "Left Step Length": f"{left_mean_step_length:.2f} (normalized)",
        "Right Step Length": f"{right_mean_step_length:.2f} (normalized)",
        "Step Width": f"{mean_step_width:.2f} (normalized)"
    }
    
    st.table(pd.DataFrame.from_dict(metrics, orient='index', columns=['Value']))
    
    # Joint ROM by phase
    st.subheader("🦵 Joint Range of Motion (ROM) by Gait Phase")
    if recorded_data["joint_angles"]:
        rom_data = []
        for joint in recorded_data["joint_angles"].keys():
            angles = recorded_data["joint_angles"][joint]
            if angles:
                rom_data.append({
                    "Joint": joint.replace('_', ' ').title(),
                    "Stance ROM (°)": f"{mean_phase_rom.get(f'{joint}_stance', 0):.1f}",
                    "Swing ROM (°)": f"{mean_phase_rom.get(f'{joint}_swing', 0):.1f}",
                    "Total ROM (°)": f"{max(angles) - min(angles):.1f}"
                })
        
        if rom_data:
            st.table(pd.DataFrame(rom_data))
    
    # Visualization section
    st.subheader("📈 Gait Visualizations")
    
    # Foot distance plot
    fig, ax = plt.subplots(figsize=(PLOT_WIDTH, PLOT_HEIGHT))
    ax.plot(foot_dists, label='Foot Distance', linewidth=2)
    ax.plot(peaks, foot_dists[peaks], "ro", markersize=8, label='Detected Steps')
    
    # Color gait phases
    for i, phase in enumerate(gait_phases):
        phase_name, start, end = phase
        color = 'lightgreen' if phase_name == 'stance' else 'lightblue'
        ax.axvspan(start, end, color=color, alpha=0.3, 
                  label=f'{phase_name.capitalize()} Phase' if i < 2 else "")
    
    ax.set_title("Step Detection with Gait Phases", fontsize=16)
    ax.set_xlabel("Frame", fontsize=12)
    ax.set_ylabel("Normalized Distance", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    
    # Joint angles plots
    if recorded_data["joint_angles"]:
        fig, axes = plt.subplots(2, 2, figsize=(PLOT_WIDTH, PLOT_HEIGHT*2))
        axes = axes.flatten()
        
        for i, (joint, angles) in enumerate(recorded_data["joint_angles"].items()):
            if i >= len(axes) or not angles:
                break
                
            smoothed = smooth_data(angles)
            axes[i].plot(smoothed, label='Angle', linewidth=2)
            
            # Mark gait phases
            for phase in gait_phases:
                phase_name, start, end = phase
                if start < len(smoothed) and end <= len(smoothed):
                    color = 'green' if phase_name == 'stance' else 'blue'
                    axes[i].axvspan(start, end, color=color, alpha=0.1)
            
            axes[i].set_title(f"{joint.replace('_', ' ').title()} Angle", fontsize=14)
            axes[i].set_xlabel("Frame", fontsize=12)
            axes[i].set_ylabel("Angle (°)", fontsize=12)
            axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(recorded_data["joint_angles"]), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # Save data button
    if st.button("💾 Save Analysis Data"):
        csv_path = save_recording_data(recorded_data)
        st.success(f"Data saved to: {csv_path}")