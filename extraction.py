import os
import json
import cv2
import mediapipe as mp
import logging

# Setup logging
LOG_FILE = "errors.log"
logging.basicConfig(filename=LOG_FILE, level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize MediaPipe Hands for landmark extraction
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Function to extract landmarks from an image
def extract_landmarks(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Image file could not be read: {image_path}")

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert image to RGB
        results = hands.process(image_rgb)  # Get hand landmarks

        landmarks = []
        num_hands_detected = 0

        if results.multi_hand_landmarks:
            num_hands_detected = len(results.multi_hand_landmarks)  # Count hands detected
            
            for hand_landmarks in results.multi_hand_landmarks:
                hand_coords = []
                for landmark in hand_landmarks.landmark:
                    hand_coords.append([landmark.x, landmark.y, landmark.z])  # Extract x, y, z
                landmarks.append(hand_coords)  # Append landmarks for each hand
        
        # If only one hand is detected, add a placeholder for the second hand (21 keypoints with 0)
        if num_hands_detected == 1:
            landmarks.append([[0, 0, 0]] * 21)  # Padding for the missing hand

        # Determine if this is a dual-hand gesture
        is_dual_hand = num_hands_detected == 2

        return {
            "landmarks": landmarks,  # List of 2 hands (or 1 + padding)
            "is_dual_hand": is_dual_hand
        }

    except Exception as e:
        logging.error(f"Error processing {image_path}: {str(e)}")
        return None

# Function to extract landmarks from a single image and save to output file
def extract_and_save_landmarks(image_path, output_path, label=None):
    try:
        print(f"Processing {image_path}...")
        landmark_data = extract_landmarks(image_path)

        if landmark_data:  # Only save if landmarks are found
            output_data = {
                "landmarks": landmark_data["landmarks"],
                "is_dual_hand": landmark_data["is_dual_hand"]
            }
            
            # Add label if provided
            if label:
                output_data["label"] = label
                
            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=4)
            print(f"Saved landmarks to {output_path}")
            return True
        else:
            print(f"No landmarks detected in {image_path}")
            return False
    except Exception as e:
        logging.error(f"Error saving landmarks for {image_path}: {str(e)}")
        return False

# Function to process a directory of images
def process_directory(input_dir, output_dir, recursive=True):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    successful = 0
    failed = 0
    
    # Process directory contents
    for item in os.listdir(input_dir):
        item_path = os.path.join(input_dir, item)
        
        # If it's a directory and recursive is True, process it recursively
        if os.path.isdir(item_path) and recursive:
            subdir_output = os.path.join(output_dir, item)
            os.makedirs(subdir_output, exist_ok=True)
            sub_success, sub_failed = process_directory(item_path, subdir_output, recursive)
            successful += sub_success
            failed += sub_failed
            
        # If it's an image, process it
        elif item.lower().endswith(('.jpg', '.jpeg', '.png')):
            output_file = os.path.join(output_dir, f"{os.path.splitext(item)[0]}.json")
            
            # Use the directory name as the label if available
            label = os.path.basename(input_dir)
            
            if extract_and_save_landmarks(item_path, output_file, label):
                successful += 1
            else:
                failed += 1
    
    return successful, failed

# Main execution function
def main():
    # Define input and output directories
    input_dir = "dataset"  # Change this to your image directory
    output_dir = "landmarks"  # Change this to your desired output directory
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Starting landmark extraction from '{input_dir}' to '{output_dir}'...")
    
    try:
        successful, failed = process_directory(input_dir, output_dir)
        print(f"Landmark extraction completed!")
        print(f"Successful extractions: {successful}")
        print(f"Failed extractions: {failed}")
    except Exception as e:
        logging.critical(f"Critical failure in landmark extraction process: {str(e)}")
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()