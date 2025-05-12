import cv2 
import mediapipe as mp 
import numpy as np 
 
mp_drawing = mp.solutions.drawing_utils 
mp_pose = mp.solutions.pose 
 
 
def calculate_angle(a, b, c): 
    a = np.array(a) 
    b = np.array(b) 
    c = np.array(c) 
 
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0]) 
    angle = np.abs(radians * 180.0 / np.pi) 
 
    if angle > 180.0: 
        angle = 360 - angle 
    return angle 
 
 
# Initialize camera settings 
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW) 
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) 
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) 
cam.set(cv2.CAP_PROP_FPS, 30) 
cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG')) 
 
# Squat counter variables 
counter = 0 
stage = None 
 
with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.5) as pose: 
    while cam.isOpened(): 
        ret, img = cam.read() 
        if not ret: 
            break 
 
        # Process image with MediaPipe 
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        results = pose.process(imgRGB) 
        img = cv2.cvtColor(imgRGB, cv2.COLOR_RGB2BGR) 
 
        try: 
            landmarks = results.pose_landmarks.landmark 
 
            # Extract LEG landmarks (left side) 
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, 
                   landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y] 
            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, 
                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y] 
            ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, 
                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y] 
 
            # Calculate knee angle 
            angle = calculate_angle(hip, knee, ankle) 
 
            # Display angle 
            cv2.putText(img, f"{angle:.1f}", 
                        tuple(np.multiply(knee, [1280, 720]).astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA) 
 
            # Squat counter logic 
            if angle > 160:  # Standing position 
                stage = "up" 
            if angle < 90 and stage == 'up':  # Squat down 
                stage = "down" 
                counter += 1 
 
        except Exception as e: 
            pass 
 
        # Display counter and stage 
        cv2.rectangle(img, (0, 0), (280, 85), (240, 100, 80), -1) 
        cv2.putText(img, 'SQUATS', (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2) 
        cv2.putText(img, str(counter), (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2) 
        cv2.putText(img, 'STAGE', (95, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2) 
        cv2.putText(img, stage, (95, 75), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2) 
 
        # Render landmarks 
        mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS) 
 
        cv2.imshow('Squat Counter (MBS3523-Q4)', img) 
 
        # Reset counter with SPACE, exit with ESC 
        key = cv2.waitKey(5) 
        if key == 32:  # SPACE 
            counter = 0 
        elif key == 27:  # ESC 
            break 
 
cam.release() 
cv2.destroyAllWindows() 

 
