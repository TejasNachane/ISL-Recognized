import cv2
import tensorflow as tf
import logging
from config import MODEL_SAVE_PATH

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def load_trained_model(model_path):
    """Load the trained model with error handling."""
    try:
        model = tf.keras.models.load_model(model_path)
        logging.info("Model loaded successfully from %s", model_path)
        return model
    except Exception as e:
        logging.error("Failed to load the model: %s", e)
        raise

def test_camera(index=0, duration=5):
    """Test the specified camera index for a short duration."""
    try:
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            logging.error("Failed to open camera at index %d.", index)
            return False
        
        logging.info("Camera %d opened successfully.", index)
        logging.info("Displaying live feed for %d seconds. Press 'q' to quit.", duration)
        
        start_time = cv2.getTickCount()
        fps = cv2.getTickFrequency()
        
        while (cv2.getTickCount() - start_time) / fps < duration:
            ret, frame = cap.read()
            if not ret:
                logging.error("Failed to capture frame from camera %d.", index)
                break
            
            cv2.imshow("Camera Test", frame)
            
            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logging.info("Exiting live feed.")
                break
        
        cap.release()
        cv2.destroyAllWindows()
        logging.info("Camera test completed successfully.")
        return True
    except Exception as e:
        logging.error("An error occurred during camera test: %s", e)
        return False

if __name__ == "__main__":
    logging.info("Starting model and camera test...")
    
    # Load model
    try:
        model = load_trained_model(MODEL_SAVE_PATH)
    except Exception as e:
        logging.critical("Critical error while loading model: %s", e)
        exit(1)
    
    # Test camera
    if not test_camera(index=0, duration=5):
        logging.critical("Camera test failed. Ensure the camera is connected and accessible.")
