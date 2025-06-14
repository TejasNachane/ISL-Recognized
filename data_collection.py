import cv2
import os
import numpy as np
import mediapipe as mp

# Directory for storing collected data
DIRECTORY = 'dataset'
if not os.path.exists(DIRECTORY):
    os.makedirs(DIRECTORY)

# Create subdirectories for each class (A-Z, 0-9, and blank)
for i in range(65, 91):  # A-Z
    letter = chr(i)
    os.makedirs(os.path.join(DIRECTORY, letter), exist_ok=True)
for i in range(10):  # 0-9
    os.makedirs(os.path.join(DIRECTORY, str(i)), exist_ok=True)
os.makedirs(os.path.join(DIRECTORY, 'blank'), exist_ok=True)

# Initialize MediaPipe Hand for landmark detection
mp_hands = mp.solutions.hands
hand_detector = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=2,  # Change to detect both hands
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Region of Interest (ROI) parameters
ROI_TOP, ROI_BOTTOM, ROI_RIGHT, ROI_LEFT = 100, 400, 150, 450

# Function for enhanced skin segmentation using YCbCr and adaptive thresholding
def enhanced_skin_segmentation(frame):
    ycbcr = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    lower_skin = np.array([0, 133, 77], dtype=np.uint8)
    upper_skin = np.array([255, 173, 127], dtype=np.uint8)
    skin_mask = cv2.inRange(ycbcr, lower_skin, upper_skin)

    # Adaptive thresholding on Y channel for better skin region isolation
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    adaptive_thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    combined_mask = cv2.bitwise_and(skin_mask, adaptive_thresh)
    segmented = cv2.bitwise_and(frame, frame, mask=combined_mask)
    return segmented

# Function for advanced hand landmark detection using MediaPipe
def detect_hand_landmarks(frame):
    # Create a copy of the frame to draw on
    frame_with_landmarks = frame.copy()
    
    # Draw ROI rectangle on the landmark frame
    cv2.rectangle(frame_with_landmarks, (ROI_LEFT, ROI_TOP), (ROI_RIGHT, ROI_BOTTOM), (255, 255, 255), 2)
    
    # Convert to RGB for MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hand_detector.process(frame_rgb)
    
    landmarks = []
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw landmarks for each detected hand
            mp.solutions.drawing_utils.draw_landmarks(frame_with_landmarks, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks.append(hand_landmarks)
    
    return frame_with_landmarks, landmarks

# Function to preprocess frames (crop, resize, normalize)
def preprocess_frame(frame):
    roi = frame[ROI_TOP:ROI_BOTTOM, ROI_RIGHT:ROI_LEFT]
    resized_frame = cv2.resize(roi, (128, 128))  # Resize to 128x128 pixels
    normalized_frame = resized_frame / 255.0  # Normalize pixel values to [0, 1]
    return normalized_frame

# Webcam initialization
cap = cv2.VideoCapture(0)
print("Press the corresponding key (A-Z, 0-9) to capture an image. Press '.' for blank. Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to access webcam.")
        break

    frame = cv2.flip(frame, 1)  # Flip the frame horizontally

    # Enhanced skin segmentation
    segmented = enhanced_skin_segmentation(frame)

    # Hand landmark detection
    frame_with_landmarks, landmarks = detect_hand_landmarks(frame)

    # Display processed frames
    cv2.imshow("Segmented", segmented)
    cv2.imshow("Hand Landmarks", frame_with_landmarks)

    # Save images based on key presses
    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF in range(ord('a'), ord('z') + 1):  # Save for A-Z
        char = chr(interrupt & 0xFF).upper()
        count = len(os.listdir(os.path.join(DIRECTORY, char)))
        roi = preprocess_frame(frame)
        save_path = os.path.join(DIRECTORY, char, f"{count}.jpg")
        cv2.imwrite(save_path, roi * 255)
        print(f"Saved: {save_path}")
    elif interrupt & 0xFF in range(ord('0'), ord('9') + 1):  # Save for 0-9
        digit = chr(interrupt & 0xFF)
        count = len(os.listdir(os.path.join(DIRECTORY, digit)))
        roi = preprocess_frame(frame)
        save_path = os.path.join(DIRECTORY, digit, f"{count}.jpg")
        cv2.imwrite(save_path, roi * 255)
        print(f"Saved: {save_path}")
    elif interrupt & 0xFF == ord('.'):  # Save for blank
        count = len(os.listdir(os.path.join(DIRECTORY, 'blank')))
        roi = preprocess_frame(frame)
        save_path = os.path.join(DIRECTORY, 'blank', f"{count}.jpg")
        cv2.imwrite(save_path, roi * 255)
        print(f"Saved: {save_path}")
    elif interrupt & 0xFF == ord('q'):  # Quit
        print("Exiting...")
        break

# Release resources
cap.release()
cv2.destroyAllWindows()