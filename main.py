import streamlit as st
import importlib.util

# Check and import cv2
if importlib.util.find_spec('cv2') is None:
    st.error("OpenCV (cv2) is not installed. Please check dependencies.")
    st.stop()
import cv2

# Check and import mediapipe (if still used)
if importlib.util.find_spec('mediapipe') is None:
    st.error("Mediapipe is not installed. Please check dependencies.")
    st.stop()
import mediapipe as mp

import time
import math
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import logging

# Initialize Mediapipe Holistic model
mp_holistic = mp.solutions.holistic
holistic_model = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Initialize drawing utilities for drawing landmarks
mp_drawing = mp.solutions.drawing_utils

# Timing variables
eye_closed_time = 0  # Time when eyes are detected closed
countdown_start_time = 0  # Start time of the countdown
countdown_active = False  # Flag for countdown status
countdown_duration = 3  # Countdown duration in seconds
action_triggered = False  # Flag to track if the action is triggered
drowsiness_counter = 0  # Drowsiness counter
max_drowsiness = 100  # Maximum value for drowsiness counter
increment_rate_eye = 10  # Increment per second when eyes closed
increment_rate_yawn = 5  # Increment per second when yawn detected
decrement_rate = 1  # Decrement per second when eyes open
ok_gesture_start_time = 0  # Time when OK gesture is detected
ok_gesture_duration = 5  # Duration OK gesture must be held (seconds)
allow_decrement = True  # Flag to control decrementing
warning_count = 0  # Counter for number of resets
previousTime = 0  # For FPS calculation

# Sensitivity variables
EAR_THRESHOLD = 0.29  # Lower value = more sensitive to eye closure
YAWN_SENSITIVITY = 0.89  # Higher value = more sensitive to yawns
BASE_YAWN_THRESHOLD = 0.4  # Base threshold for yawn detection

# Indices for left and right eyes (from Mediapipe)
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144, 163, 144, 145, 154]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380, 384, 385, 386, 367]

def calculate_mouth_aspect_ratio(landmarks):
    """Calculate the mouth aspect ratio (MAR) to detect yawn."""
    top_lip = landmarks[13]  # Upper lip, just below nose
    bottom_lip = landmarks[14]  # Lower lip, just above chin
    left_corner = landmarks[78]  # Left mouth corner
    right_corner = landmarks[308]  # Right mouth corner

    horizontal_distance = math.dist((left_corner.x, left_corner.y), (right_corner.x, right_corner.y))
    vertical_distance = math.dist((top_lip.x, top_lip.y), (bottom_lip.x, bottom_lip.y))

    return vertical_distance / horizontal_distance

def calculate_eye_aspect_ratio(landmarks, eye_indices):
    """Calculate the eye aspect ratio (EAR) to detect eye open or close."""
    eye = [landmarks[i] for i in eye_indices]
    horizontal_distance = math.dist((eye[0].x, eye[0].y), (eye[3].x, eye[3].y))
    vertical_distance_1 = math.dist((eye[1].x, eye[1].y), (eye[5].x, eye[5].y))
    vertical_distance_2 = math.dist((eye[2].x, eye[2].y), (eye[4].x, eye[4].y))
    ear = (vertical_distance_1 + vertical_distance_2) / (2.0 * horizontal_distance)
    return ear

def detect_ok_gesture(hand_landmarks):
    """Detect OK gesture (thumb touching index finger tip, other fingers extended)."""
    if not hand_landmarks:
        return False
    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]
    middle_tip = hand_landmarks.landmark[12]
    ring_tip = hand_landmarks.landmark[16]
    pinky_tip = hand_landmarks.landmark[20]
    wrist = hand_landmarks.landmark[0]
    thumb_index_dist = math.dist((thumb_tip.x, thumb_tip.y), (index_tip.x, index_tip.y))
    middle_wrist_dist = math.dist((middle_tip.x, middle_tip.y), (wrist.x, wrist.y))
    ring_wrist_dist = math.dist((ring_tip.x, ring_tip.y), (wrist.x, wrist.y))
    pinky_wrist_dist = math.dist((pinky_tip.x, pinky_tip.y), (wrist.x, wrist.y))
    return (thumb_index_dist < 0.05 and
            middle_wrist_dist > 0.2 and
            ring_wrist_dist > 0.2 and
            pinky_wrist_dist > 0.2)

# Streamlit WebRTC video transformer
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.previousTime = 0
        self.eye_closed_time = 0
        self.countdown_start_time = 0
        self.countdown_active = False
        self.action_triggered = False
        self.drowsiness_counter = 0
        self.ok_gesture_start_time = 0
        self.allow_decrement = True
        self.warning_count = 0

    def transform(self, frame):
        global eye_closed_time, countdown_start_time, countdown_active, action_triggered
        global drowsiness_counter, ok_gesture_start_time, allow_decrement, warning_count
        global previousTime

        img = frame.to_ndarray(format="bgr24")
        img = cv2.resize(img, (800, 600))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb.flags.writeable = False
        results = holistic_model.process(img_rgb)
        img_rgb.flags.writeable = True
        img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        currentTime = time.time()
        delta_time = currentTime - self.previousTime if self.previousTime != 0 else 0.033  # Assume ~30 FPS if no previous time
        self.previousTime = currentTime
        fps = 1 / delta_time if delta_time > 0 else 0

        if not results.face_landmarks:
            cv2.putText(img, "No User Detected", (10, 180), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        else:
            mp_drawing.draw_landmarks(
                img,
                results.face_landmarks,
                mp_holistic.FACEMESH_CONTOURS,
                mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=1)
            )

            mouth_aspect_ratio = calculate_mouth_aspect_ratio(results.face_landmarks.landmark)
            yawn_detected = mouth_aspect_ratio > (BASE_YAWN_THRESHOLD / YAWN_SENSITIVITY)
            if yawn_detected:
                cv2.putText(img, "Yawn Detected!", (10, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                self.drowsiness_counter = min(max_drowsiness, self.drowsiness_counter + increment_rate_yawn * delta_time)

            left_eye_ear = calculate_eye_aspect_ratio(results.face_landmarks.landmark, LEFT_EYE_INDICES)
            right_eye_ear = calculate_eye_aspect_ratio(results.face_landmarks.landmark, RIGHT_EYE_INDICES)
            eyes_closed = left_eye_ear < EAR_THRESHOLD or right_eye_ear < EAR_THRESHOLD
            if eyes_closed:
                if self.eye_closed_time == 0:
                    self.eye_closed_time = time.time()
                self.drowsiness_counter = min(max_drowsiness, self.drowsiness_counter + increment_rate_eye * delta_time)
                if time.time() - self.eye_closed_time > 1.5 and not self.countdown_active:
                    self.countdown_start_time = time.time()
                    self.countdown_active = True
            else:
                self.eye_closed_time = 0
                self.countdown_active = False
                if self.allow_decrement and self.drowsiness_counter > 0:
                    self.drowsiness_counter = max(0, self.drowsiness_counter - decrement_rate * delta_time)

            if self.drowsiness_counter >= max_drowsiness:
                self.allow_decrement = False

            if self.countdown_active:
                elapsed_time = time.time() - self.countdown_start_time
                remaining_time = max(0, countdown_duration - elapsed_time)
                cv2.putText(img, f"Countdown: {int(remaining_time)}s", (10, 150), cv2.FONT_HERSHEY_COMPLEX, 1,
                            (0, 0, 255), 2)
                if remaining_time == 0:
                    if eyes_closed:
                        if not self.action_triggered:
                            print("Action Triggered!")
                            self.action_triggered = True
                    self.countdown_active = False
            else:
                self.action_triggered = False

            counter_color = (0, 255, 0)
            if self.drowsiness_counter >= max_drowsiness:
                counter_color = (0, 0, 255)
                cv2.putText(img, "CAUTION: WHEELS STOPPED. DRIVING NOT ADVISED", (10, 180),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            elif self.drowsiness_counter > 75:
                counter_color = (0, 0, 255)
                cv2.putText(img, "CAUTION: PULL OVER", (10, 180), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            elif self.drowsiness_counter > 50:
                counter_color = (0, 255, 255)

            cv2.putText(img, f"Drowsiness: {int(self.drowsiness_counter)}", (550, 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, counter_color, 2)

        if self.drowsiness_counter >= max_drowsiness:
            ok_detected = (detect_ok_gesture(results.left_hand_landmarks) or
                          detect_ok_gesture(results.right_hand_landmarks))
            if ok_detected:
                if self.ok_gesture_start_time == 0:
                    self.ok_gesture_start_time = time.time()
                elif time.time() - self.ok_gesture_start_time >= ok_gesture_duration:
                    self.warning_count += 1
                    self.drowsiness_counter = 0
                    self.allow_decrement = True
                    self.ok_gesture_start_time = 0
                    cv2.putText(img, f"Warning {self.warning_count}", (10, 210), cv2.FONT_HERSHEY_COMPLEX, 1,
                                (0, 255, 0), 2)
            else:
                self.ok_gesture_start_time = 0
            if ok_detected:
                cv2.putText(img, "OK Gesture Detected", (10, 240), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(img, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(img, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        cv2.putText(img, f"{int(fps)} FPS", (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

        return img

# Streamlit app
st.title("DrowsyGuard")

# Startup screen
startup_placeholder = st.empty()
start_time = time.time()
startup_duration = 2
while time.time() - start_time < startup_duration:
    startup_frame = np.zeros((600, 800, 3), dtype=np.uint8)
    cv2.putText(startup_frame, "DrowsyGuard", (250, 300), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 3)
    progress = min(100, ((time.time() - start_time) / startup_duration) * 100)
    cv2.putText(startup_frame, f"Initializing: {int(progress)}%", (10, 570), cv2.FONT_HERSHEY_COMPLEX, 1,
                (255, 255, 255), 2)
    startup_placeholder.image(startup_frame, channels="BGR")
    time.sleep(0.1)
startup_placeholder.empty()

# Mesh alignment screen
mesh_placeholder = st.empty()
mesh_start_time = time.time()
mesh_duration = 2
while time.time() - mesh_start_time < mesh_duration:
    mesh_frame = np.zeros((600, 800, 3), dtype=np.uint8)  # Placeholder since we can't access webcam yet
    cv2.putText(mesh_frame, "Generating User Sitting", (200, 280), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(mesh_frame, "and Alignment Mesh", (250, 320), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
    mesh_placeholder.image(mesh_frame, channels="BGR")
    time.sleep(0.1)
mesh_placeholder.empty()

# WebRTC streamer for real-time video processing
RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
webrtc_streamer(key="drowsyguard", video_transformer_factory=VideoTransformer, rtc_configuration=RTC_CONFIGURATION)