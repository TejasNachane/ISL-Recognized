from flask import Flask, render_template, Response, request, jsonify, send_from_directory
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import json
import mediapipe as mp
import threading
import time
import pyttsx3
import logging
import queue
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

# Initialize Flask app
app = Flask(__name__)

# Initialize the video capture outside the class for better Flask compatibility
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    logging.error("Could not open camera")

# Prediction flags and current prediction
prediction_active = False
last_prediction_time = time.time()
prediction_delay = 1  # seconds between predictions
confidence_threshold = 0.7  # Minimum confidence required for a prediction

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
        
        # Initialize prediction history for smoothing
        self.prediction_history = deque(maxlen=5)
        
        # Properties for controlling the UI
        self.recognized_text = ""
        self.current_prediction = "No sign detected"
        self.prediction_confidence = 0.0
        self.max_chars_per_line = 30
            
    def _load_model(self, model_path):
        """Load the trained model"""
        try:
            if not os.path.exists(model_path):
                logging.error(f"Model file not found: {model_path}")
                # Try default path
                if os.path.exists("model.h5"):
                    model_path = "model.h5"
                    logging.warning(f"Using default model path: {model_path}")
                else:
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
                # Try default path
                if os.path.exists("label_encoder.json"):
                    label_path = "label_encoder.json"
                    logging.warning(f"Using default label path: {label_path}")
                else:
                    raise FileNotFoundError(f"Label mapping file not found: {label_path}")
                
            with open(label_path, "r") as f:
                label_mapping = json.load(f)
            
            # Handle both formats - direct mapping or indexed mapping
            if isinstance(label_mapping, dict) and all(k.isdigit() for k in label_mapping.keys()):
                # Format from predict.py
                classes = [label_mapping[str(i)] for i in sorted([int(k) for k in label_mapping.keys()])]
            else:
                # Format from app.py
                classes = list(label_mapping.values())
                
            logging.info(f"Label encoder loaded with classes: {classes}")
            return classes
        except Exception as e:
            logging.error(f"Error loading label encoder: {e}")
            raise
            
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
    
    def update_accumulated_text(self, new_prediction):
        """Add a new line to accumulated text if needed"""
        # Find the last newline character
        last_newline_pos = self.recognized_text.rfind('\n')
        
        # Calculate the length of the current line
        if last_newline_pos == -1:
            current_line_length = len(self.recognized_text)
        else:
            current_line_length = len(self.recognized_text) - last_newline_pos - 1
        
        # If adding the new prediction would exceed the max chars per line, add a newline
        if current_line_length + len(new_prediction) > self.max_chars_per_line:
            self.recognized_text += '\n' + new_prediction
        else:
            self.recognized_text += new_prediction
            
    def reset_text(self):
        """Reset accumulated text"""
        self.recognized_text = ""
        return {"status": "success", "message": "Text reset successfully"}
        
    def save_text(self):
        """Save accumulated text to file"""
        try:
            with open("recognized_text.txt", "w") as f:
                f.write(self.recognized_text)
            return {"status": "success", "message": "Text saved successfully", "text": self.recognized_text}
        except Exception as e:
            logging.error(f"Error saving text: {e}")
            return {"status": "error", "message": f"Error saving text: {str(e)}"}
            
# Speech queue and thread control for TTS
speech_queue = queue.Queue()
speech_thread_running = True

# Text-to-speech function that runs in a dedicated thread
def speech_worker():
    # Initialize the text-to-speech engine inside the thread
    tts_engine = pyttsx3.init()
    
    # Configure the engine
    tts_engine.setProperty('rate', 150)  # Speed - words per minute
    tts_engine.setProperty('volume', 0.9)  # Volume (0-1)
    
    # Get available voices and set to a clearer one if available
    voices = tts_engine.getProperty('voices')
    if len(voices) > 0:
        tts_engine.setProperty('voice', voices[0].id)  # Use the first available voice
    
    while speech_thread_running:
        try:
            # Get text from queue with a timeout to allow checking thread_running
            text = speech_queue.get(timeout=0.5)
            if text:
                tts_engine.say(text)
                tts_engine.runAndWait()
            speech_queue.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            logging.error(f"Error in speech worker: {e}")
            time.sleep(1)  # Wait a bit before retrying

# Start the speech worker thread
speech_thread = threading.Thread(target=speech_worker, daemon=True)
speech_thread.start()

def speak_text_async(text):
    """
    Add text to the speech queue for asynchronous vocalization.
    """
    try:
        speech_queue.put(text)
    except Exception as e:
        logging.error(f"Error adding text to speech queue: {e}")

# Initialize gesture recognizer
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
    
    # Create the gesture recognizer
    recognizer = GestureRecognizer(model_path, label_path)
    
except Exception as e:
    logging.error(f"Error initializing gesture recognizer: {e}")
    exit()

# Function to generate frames for video streaming - MODIFIED to remove prediction display
def generate_frames():
    global prediction_active, last_prediction_time
    
    while True:
        success, frame = camera.read()
        if not success:
            logging.error("Failed to capture frame")
            break
        else:
            # Flip frame horizontally for a mirrored view
            frame = cv2.flip(frame, 1)
            
            # Perform prediction only if active
            if prediction_active:
                landmarks_list, frame_with_landmarks = recognizer.extract_landmarks_from_frame(frame)
                
                if landmarks_list:
                    current_time = time.time()
                    
                    # Check if enough time has passed since the last prediction
                    if current_time - last_prediction_time >= prediction_delay:
                        try:
                            # Prepare input and make prediction
                            input_features = recognizer.prepare_input_features(landmarks_list)
                            prediction, confidence = recognizer.predict_gesture(input_features)
                            
                            if prediction is not None and confidence >= confidence_threshold:
                                # Update the current prediction
                                recognizer.current_prediction = prediction
                                recognizer.prediction_confidence = confidence
                                
                                # Append the predicted label to the accumulated text with line breaks
                                recognizer.update_accumulated_text(prediction)
                                
                                # Speak the predicted label asynchronously
                                speak_text_async(prediction)
                            
                            # Update the last prediction time
                            last_prediction_time = current_time
                        except Exception as e:
                            logging.error(f"Error during prediction: {e}")
                else:
                    recognizer.current_prediction = "No hand detected"
                    frame_with_landmarks = frame
            else:
                recognizer.current_prediction = "Detection paused"
                frame_with_landmarks = frame

            # Encode the frame and return it as a response
            # We are only using the raw frame_with_landmarks without drawing UI elements
            _, buffer = cv2.imencode('.jpg', frame_with_landmarks)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# Flask route to serve the homepage
@app.route('/')
def index():
    return render_template('index.html')

# Flask route for video streaming
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Flask route to toggle prediction state
@app.route('/toggle_prediction', methods=['POST'])
def toggle_prediction():
    global prediction_active
    data = request.json
    prediction_active = data['active']
    return jsonify({"status": "success", "prediction_active": prediction_active})

# Server-Sent Events endpoint for prediction updates
@app.route('/prediction_stream')
def prediction_stream():
    def generate():
        last_prediction = None
        last_text = None
        
        while True:
            # Send an update when either prediction or accumulated text changes
            if recognizer.current_prediction != last_prediction or recognizer.recognized_text != last_text:
                last_prediction = recognizer.current_prediction
                last_text = recognizer.recognized_text
                data = {
                    "prediction": last_prediction,
                    "accumulatedText": last_text,
                    "confidence": f"{recognizer.prediction_confidence:.2f}"
                }
                yield f"data: {json.dumps(data)}\n\n"
            
            time.sleep(0.1)  # Check for updates 10 times per second
    
    return Response(generate(), mimetype="text/event-stream")

# Reset accumulated text
@app.route('/reset_text', methods=['POST'])
def reset_text():
    result = recognizer.reset_text()
    return jsonify(result)

# Save accumulated text
@app.route('/save_text', methods=['GET'])
def save_text():
    result = recognizer.save_text()
    return jsonify(result)

# Serve dataset images
DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset')

@app.route('/dataset/<path:filename>')
def serve_dataset_images(filename):
    """
    Custom route to serve images from the dataset directory
    Usage in HTML: <img src="/dataset/A/1.jpg">
    """
    return send_from_directory(DATASET_PATH, filename)

# Clean up resources when the application exits
def cleanup():
    global speech_thread_running
    speech_thread_running = False
    if speech_thread.is_alive():
        speech_thread.join(timeout=1)
    camera.release()
    recognizer.hands_processor.close()

# Register cleanup function to be called when Flask shuts down
import atexit
atexit.register(cleanup)

# Main entry point
if __name__ == '__main__':
    app.run(debug=True, threaded=True)