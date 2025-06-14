import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import json
import mediapipe as mp
import pyttsx3
import threading
import time
import logging
import os
from collections import deque

# Initialize logging with more detailed configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("gesture_recognition.log"),
        logging.StreamHandler()
    ]
)

class GestureRecognizer:
    """Class to handle real-time hand gesture recognition"""
    
    def __init__(self, model_path="models/hand_gesture_classifier.h5", label_path="label_mapping.json"):
        """Initialize the gesture recognizer with model and label encoder"""
        self.model = self._load_model(model_path)
        self.label_encoder = self._load_label_encoder(label_path)
        
        # Initialize MediaPipe Hands processor
        self.mp_hands = mp.solutions.hands
        self.hands_processor = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize text-to-speech engine
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 150)  # Adjust speaking rate
        
        # Initialize prediction history for smoothing
        self.prediction_history = deque(maxlen=5)
        
        # Properties for controlling the UI
        self.recognized_text = ""
        self.current_prediction = ""
        self.prediction_confidence = 0.0
        
        # Flag to control speech output
        self.speech_enabled = True
        
    def _load_model(self, model_path):
        """Load the trained model"""
        try:
            if not os.path.exists(model_path):
                logging.error(f"Model file not found: {model_path}")
                raise FileNotFoundError(f"Model file not found: {model_path}")
                
            model = load_model(model_path)
            logging.info(f"Model loaded successfully from {model_path}")
            return model
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise
            
    def _load_label_encoder(self, label_path):
        """Load the label encoder mapping"""
        try:
            if not os.path.exists(label_path):
                logging.error(f"Label mapping file not found: {label_path}")
                raise FileNotFoundError(f"Label mapping file not found: {label_path}")
                
            with open(label_path, "r") as f:
                label_mapping = json.load(f)
                
            # Convert the dictionary to the format expected by LabelEncoder
            classes = [label_mapping[str(i)] for i in sorted([int(k) for k in label_mapping.keys()])]
            logging.info(f"Label encoder loaded with classes: {classes}")
            return classes
        except Exception as e:
            logging.error(f"Error loading label encoder: {e}")
            raise
            
    def speak_text_async(self, text):
        """Speak text asynchronously"""
        if self.speech_enabled:
            threading.Thread(target=self._speak_text, args=(text,), daemon=True).start()
            
    def _speak_text(self, text):
        """Speak the given text"""
        try:
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        except Exception as e:
            logging.error(f"Speech error: {e}")
            
    def extract_landmarks_from_frame(self, frame):
        """Extract hand landmarks from a video frame"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands_processor.process(frame_rgb)
        
        # Copy frame for visualization
        annotated_frame = frame.copy()
        
        # Data to store landmarks for both hands
        all_landmarks = []
        
        if results.multi_hand_landmarks:
            num_hands = len(results.multi_hand_landmarks)
            
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Draw landmarks on the frame
                self.mp_drawing.draw_landmarks(
                    annotated_frame, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                    self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                )
                
                # Extract landmark positions (x, y, z)
                hand_points = []
                for landmark in hand_landmarks.landmark:
                    hand_points.append([landmark.x, landmark.y, landmark.z])
                
                all_landmarks.append(hand_points)
            
            # Ensure we have data for exactly 2 hands (add zero padding if needed)
            if num_hands == 1:
                # Add zero padding for the missing hand
                all_landmarks.append([[0, 0, 0] for _ in range(21)])
                
        return all_landmarks, annotated_frame
        
    def prepare_input_features(self, landmarks):
        """Prepare the input features for the model"""
        if not landmarks:
            return None
            
        # Flatten the structure to match training input format
        feature_vector = []
        for hand in landmarks:
            for landmark in hand:
                feature_vector.extend(landmark)  # Add x, y, z
                
        return np.array([feature_vector], dtype=np.float32)
        
    def predict_gesture(self, input_features):
        """Make a prediction using the model"""
        if input_features is None:
            return None, 0.0
            
        try:
            # Get model predictions
            predictions = self.model.predict(input_features, verbose=0)
            
            # Get the predicted class index and confidence
            predicted_idx = np.argmax(predictions[0])
            confidence = predictions[0][predicted_idx]
            
            # Convert to label
            predicted_label = self.label_encoder[predicted_idx]
            
            # Update prediction history for smoothing
            self.prediction_history.append(predicted_label)
            
            # Return the most common prediction in the history
            from collections import Counter
            most_common = Counter(self.prediction_history).most_common(1)
            smoothed_prediction = most_common[0][0] if most_common else predicted_label
            
            return smoothed_prediction, confidence
        except Exception as e:
            logging.error(f"Prediction error: {e}")
            return None, 0.0
            
    def run_realtime_detection(self, camera_id=0, prediction_interval=10, prediction_delay=1.5, confidence_threshold=0.7):
        """Run real-time detection using webcam
        
        Args:
            camera_id: Camera device ID (default: 0)
            prediction_interval: Number of frames between prediction attempts (default: 10)
            prediction_delay: Minimum time in seconds between predictions (default: 1.5)
            confidence_threshold: Minimum confidence required for a prediction (default: 0.7)
        """
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            logging.error(f"Could not open camera {camera_id}")
            return
            
        logging.info("Starting real-time detection. Controls:")
        logging.info("  'q': Quit the application")
        logging.info("  'r': Reset recognized text")
        logging.info("  's': Save recognized text to file")
        logging.info("  'm': Toggle speech on/off")
        
        frame_count = 0
        last_prediction_time = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                logging.error("Failed to capture frame")
                break
                
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Process frame to get landmarks
            landmarks, annotated_frame = self.extract_landmarks_from_frame(frame)
            
            # Make predictions at specified intervals and only after delay time has passed
            current_time = time.time()
            make_prediction = (
                landmarks and 
                frame_count % prediction_interval == 0 and 
                current_time - last_prediction_time >= prediction_delay
            )
            
            if make_prediction:
                # Prepare input and make prediction
                input_features = self.prepare_input_features(landmarks)
                prediction, confidence = self.predict_gesture(input_features)
                
                if prediction is not None and confidence >= confidence_threshold:
                    self.current_prediction = prediction
                    self.prediction_confidence = confidence
                    
                    # Add to recognized text
                    self.recognized_text += prediction
                    
                    # Speak the prediction
                    self.speak_text_async(prediction)
                    
                last_prediction_time = current_time
            
            # Display UI elements
            self._draw_ui(annotated_frame, landmarks)
            
            # Show the frame
            cv2.imshow("Hand Gesture Recognition", annotated_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):  # Quit
                break
            elif key == ord("r"):  # Reset text
                self.recognized_text = ""
                logging.info("Recognized text has been reset")
            elif key == ord("s"):  # Save text to file
                self._save_recognized_text()
            elif key == ord("m"):  # Toggle speech
                self.speech_enabled = not self.speech_enabled
                logging.info(f"Speech {'enabled' if self.speech_enabled else 'disabled'}")
            
            frame_count += 1
            
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        self.hands_processor.close()
        
    def _draw_ui(self, frame, landmarks):
        """Draw UI elements on the frame"""
        h, w, _ = frame.shape
        
        # Draw background rectangle for text
        cv2.rectangle(frame, (0, 0), (w, 140), (255, 255, 255), -1)
        cv2.rectangle(frame, (0, 0), (w, 140), (0, 0, 0), 2)
        
        # No hands detected message
        if not landmarks:
            cv2.putText(
                frame,
                "No hands detected",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )
        else:
            # Current prediction with confidence
            cv2.putText(
                frame,
                f"Gesture: {self.current_prediction} ({self.prediction_confidence:.2f})",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 128, 0),
                2,
                cv2.LINE_AA
            )
            
        # Recognized text
        text_to_display = self.recognized_text
        if len(text_to_display) > 30:  # Show only the last 30 chars
            text_to_display = "..." + text_to_display[-30:]
            
        cv2.putText(
            frame,
            f"Text: {text_to_display}",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2,
            cv2.LINE_AA
        )
        
        # Speech status
        speech_status = "ON" if self.speech_enabled else "OFF"
        cv2.putText(
            frame,
            f"Speech: {speech_status}",
            (20, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255) if not self.speech_enabled else (0, 128, 0),
            2,
            cv2.LINE_AA
        )
            
    def _save_recognized_text(self):
        """Save recognized text to a file"""
        try:
            with open("recognized_text.txt", "w") as f:
                f.write(self.recognized_text)
            logging.info("Recognized text saved to 'recognized_text.txt'")
        except Exception as e:
            logging.error(f"Error saving text: {e}")


def main():
    """Main function to run the application"""
    try:
        # Set up paths
        model_path = "models/hand_gesture_classifier.h5"
        label_path = "label_mapping.json"
        
        # Check if files exist, if not, use defaults
        if not os.path.exists(model_path):
            model_path = "model.h5"
            logging.warning(f"Using default model path: {model_path}")
            
        if not os.path.exists(label_path):
            label_path = "label_encoder.json"
            logging.warning(f"Using default label path: {label_path}")
        
        # Create and run the gesture recognizer
        recognizer = GestureRecognizer(model_path, label_path)
        
        # Use the same parameters as the original script
        recognizer.run_realtime_detection(
            camera_id=0,
            prediction_interval=10,  # Check every 10 frames
            prediction_delay=1,    # Wait 1.5 seconds between predictions
            confidence_threshold=0.7
        )
    except Exception as e:
        logging.error(f"Application error: {e}")
        print(f"Error: {str(e)}. See log file for details.")


if __name__ == "__main__":
    main()