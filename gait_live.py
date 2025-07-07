def run_live_gait_analysis():
    import cv2
    import mediapipe as mp
    import csv, os
    from datetime import datetime

    os.makedirs("outputs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = f"outputs/gait_live_{timestamp}.csv"
    video_file = f"outputs/gait_live_{timestamp}.mp4"

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    width, height = int(cap.get(3)), int(cap.get(4))
    out = cv2.VideoWriter(video_file, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

    landmark_names = [l.name for l in mp_pose.PoseLandmark]
    header = ['frame'] + [f"{n}_{a}" for n in landmark_names for a in ['x', 'y', 'z', 'visibility']]
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)
            if result.pose_landmarks:
                mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                row = [frame_idx] + [getattr(lm, attr) for lm in result.pose_landmarks.landmark for attr in ['x', 'y', 'z', 'visibility']]
                writer.writerow(row)
            out.write(frame)
            cv2.imshow('Live Gait', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            frame_idx += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()