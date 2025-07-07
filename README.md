# 🧠 Gait Analysis with MediaPipe & Streamlit

This repository enables **automatic gait analysis** using **MediaPipe Pose Estimation** from:
- 🟢 **Live webcam input**, or
- 📼 **Pre-recorded video files** stored in the `videos/` folder.

The system uses **MediaPipe** to extract 33 body landmarks per frame and outputs:
- A processed video with pose overlays
- A `.csv` file with normalized landmark coordinates

---

## 📁 Folder Structure
gait_analysis_mediapipe/
│
├── app.py                   # Streamlit web UI
├── gait_live.py             # Capture gait from webcam
├── gait_from_video.py       # Process video file
├── requirements.txt         # Dependencies
│
├── videos/                  # Upload your raw gait videos here (.mp4)
│   └── gait_20250707_120300.mp4
│
├── outputs/                 # Results: processed videos and landmark CSVs
│   ├── gait_processed_.mp4
│   └── gait_landmarks_.csv

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/GiorgosBouh/gait_analysis_mediapipe.git
cd gait_analysis_mediapipe

2. Install dependencies
pip install -r requirements.txt

3. Launch the app
streamlit run app.py

Usage

Option 1: Live Camera Gait Analysis
	•	Select “Live Camera” in the Streamlit UI.
	•	Click “Start Live Capture”.
	•	Walk in front of your camera.
	•	Press q to stop recording.

Option 2: Video File Gait Analysis
	•	Place your .mp4 video in the videos/ folder.
	•	Make sure the filename follows:
gait_YYYYMMDD_HHMMSS.mp4
(e.g., gait_20250707_121300.mp4)
	•	Select the video from the dropdown menu in the app.
	•	Click “Run analysis on selected video”.

⸻

📤 Outputs

All results are saved in the outputs/ folder:
	•	gait_processed_*.mp4: Video with pose drawn
	•	gait_landmarks_*.csv: Landmark coordinates per frame

Requirements
	•	Python 3.7+
	•	OpenCV
	•	MediaPipe
	•	Streamlit
 Notes
	•	Use side view (sagittal plane) for best gait results.
	•	For research or ML purposes, you can post-process the CSVs to calculate:
	•	Step length
	•	Cadence
	•	Symmetry indices
	•	Joint angles (e.g., hip/knee)

⸻

👨‍💻 Author

Built by Giorgos Bouh
🏫 Metropolitan College, Department of Sports & Physical Education
💡 Biomechanics & AI Researcher
