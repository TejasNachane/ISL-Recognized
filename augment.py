import os
import json
import numpy as np
import random
import logging
from tqdm import tqdm
from copy import deepcopy

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class LandmarkAugmenter:
    """
    Class for augmenting hand landmark data.
    Supports various transformations applicable to coordinate data.
    """
    
    def __init__(self, input_dir, output_dir, augmentations_per_sample=5):
        """
        Initialize the augmenter.
        
        Args:
            input_dir: Directory containing the original landmark JSON files
            output_dir: Directory to save augmented landmark JSON files
            augmentations_per_sample: Number of augmented samples to generate per original sample
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.augmentations_per_sample = augmentations_per_sample
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
    
    def _apply_noise(self, landmarks, noise_level=0.01):
        """
        Add random noise to landmarks.
        
        Args:
            landmarks: List of hand landmarks
            noise_level: Maximum magnitude of noise to add (relative to coordinate scale)
            
        Returns:
            Augmented landmarks
        """
        augmented = deepcopy(landmarks)
        
        for hand_idx in range(len(augmented)):
            if not any(augmented[hand_idx][0]):  # Skip if this is a zero-padded hand
                continue
                
            for landmark_idx in range(len(augmented[hand_idx])):
                # Add random noise to each coordinate
                noise = np.random.uniform(-noise_level, noise_level, size=3)
                augmented[hand_idx][landmark_idx][0] += noise[0]
                augmented[hand_idx][landmark_idx][1] += noise[1]
                augmented[hand_idx][landmark_idx][2] += noise[2]
                
        return augmented
    
    def _apply_rotation(self, landmarks, max_angle=15):
        """
        Apply a small random rotation to the landmarks.
        Note: This is a simplified 2D rotation that affects x and y coordinates.
        
        Args:
            landmarks: List of hand landmarks
            max_angle: Maximum rotation angle in degrees
            
        Returns:
            Augmented landmarks
        """
        augmented = deepcopy(landmarks)
        angle = np.random.uniform(-max_angle, max_angle)
        angle_rad = np.radians(angle)
        
        for hand_idx in range(len(augmented)):
            if not any(augmented[hand_idx][0]):  # Skip if this is a zero-padded hand
                continue
            
            # Find hand center for rotation reference
            x_coords = [lm[0] for lm in augmented[hand_idx]]
            y_coords = [lm[1] for lm in augmented[hand_idx]]
            center_x = np.mean(x_coords)
            center_y = np.mean(y_coords)
            
            # Rotate around the center
            for landmark_idx in range(len(augmented[hand_idx])):
                x = augmented[hand_idx][landmark_idx][0] - center_x
                y = augmented[hand_idx][landmark_idx][1] - center_y
                
                # Apply rotation
                new_x = x * np.cos(angle_rad) - y * np.sin(angle_rad)
                new_y = x * np.sin(angle_rad) + y * np.cos(angle_rad)
                
                # Update coordinates
                augmented[hand_idx][landmark_idx][0] = new_x + center_x
                augmented[hand_idx][landmark_idx][1] = new_y + center_y
                
        return augmented
    
    def _apply_scaling(self, landmarks, scale_range=(0.9, 1.1)):
        """
        Apply random scaling to landmarks.
        
        Args:
            landmarks: List of hand landmarks
            scale_range: Tuple of (min_scale, max_scale)
            
        Returns:
            Augmented landmarks
        """
        augmented = deepcopy(landmarks)
        scale = np.random.uniform(scale_range[0], scale_range[1])
        
        for hand_idx in range(len(augmented)):
            if not any(augmented[hand_idx][0]):  # Skip if this is a zero-padded hand
                continue
            
            # Find hand center for scaling reference
            x_coords = [lm[0] for lm in augmented[hand_idx]]
            y_coords = [lm[1] for lm in augmented[hand_idx]]
            z_coords = [lm[2] for lm in augmented[hand_idx]]
            center_x = np.mean(x_coords)
            center_y = np.mean(y_coords)
            center_z = np.mean(z_coords)
            
            # Scale around the center
            for landmark_idx in range(len(augmented[hand_idx])):
                augmented[hand_idx][landmark_idx][0] = center_x + (augmented[hand_idx][landmark_idx][0] - center_x) * scale
                augmented[hand_idx][landmark_idx][1] = center_y + (augmented[hand_idx][landmark_idx][1] - center_y) * scale
                augmented[hand_idx][landmark_idx][2] = center_z + (augmented[hand_idx][landmark_idx][2] - center_z) * scale
                
        return augmented
    
    def _apply_translation(self, landmarks, max_shift=0.05):
        """
        Apply random translation to landmarks.
        
        Args:
            landmarks: List of hand landmarks
            max_shift: Maximum shift as a fraction of the coordinate range
            
        Returns:
            Augmented landmarks
        """
        augmented = deepcopy(landmarks)
        
        # Generate random shifts for x, y, z
        shift_x = np.random.uniform(-max_shift, max_shift)
        shift_y = np.random.uniform(-max_shift, max_shift)
        shift_z = np.random.uniform(-max_shift, max_shift)
        
        for hand_idx in range(len(augmented)):
            if not any(augmented[hand_idx][0]):  # Skip if this is a zero-padded hand
                continue
                
            for landmark_idx in range(len(augmented[hand_idx])):
                augmented[hand_idx][landmark_idx][0] += shift_x
                augmented[hand_idx][landmark_idx][1] += shift_y
                augmented[hand_idx][landmark_idx][2] += shift_z
                
        return augmented
    
    def _apply_finger_variation(self, landmarks, max_variation=0.03):
        """
        Apply random variations to finger joints to simulate different finger positions.
        
        Args:
            landmarks: List of hand landmarks
            max_variation: Maximum variation to apply to finger joints
            
        Returns:
            Augmented landmarks
        """
        augmented = deepcopy(landmarks)
        
        # Define finger joint indices (MediaPipe hand model)
        # Each finger has 4 landmarks: base, 1st joint, 2nd joint, fingertip
        fingers = {
            'thumb': [1, 2, 3, 4],
            'index': [5, 6, 7, 8],
            'middle': [9, 10, 11, 12],
            'ring': [13, 14, 15, 16],
            'pinky': [17, 18, 19, 20]
        }
        
        for hand_idx in range(len(augmented)):
            if not any(augmented[hand_idx][0]):  # Skip if this is a zero-padded hand
                continue
            
            # Apply random variations to each finger
            for finger in fingers.values():
                # Skip base joints (keep them more stable)
                for i in range(1, len(finger)):
                    # Generate random variation
                    variation = np.random.uniform(-max_variation, max_variation, size=3)
                    
                    # Apply variation to joint
                    landmark_idx = finger[i]
                    augmented[hand_idx][landmark_idx][0] += variation[0]
                    augmented[hand_idx][landmark_idx][1] += variation[1]
                    augmented[hand_idx][landmark_idx][2] += variation[2]
                    
        return augmented
    
    def augment_file(self, json_path):
        """
        Augment a single landmark JSON file.
        
        Args:
            json_path: Path to the JSON file containing landmark data
            
        Returns:
            List of augmented landmark data dictionaries
        """
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                
            if not data.get('landmarks'):
                logging.warning(f"No landmarks found in {json_path}, skipping.")
                return []
            
            # Get original landmarks
            original_landmarks = data['landmarks']
            
            # Extract label from filename or data
            label = data.get('label')
            if not label:
                # Try to get label from directory structure
                parent_dir = os.path.basename(os.path.dirname(json_path))
                if parent_dir != os.path.basename(self.input_dir):
                    label = parent_dir
                else:
                    # If no label available, use filename without extension
                    label = os.path.splitext(os.path.basename(json_path))[0]
            
            is_dual_hand = data.get('is_dual_hand', False)
            
            augmented_samples = []
            
            # Create multiple augmented versions
            for i in range(self.augmentations_per_sample):
                augmented_landmarks = original_landmarks
                
                # Apply a random combination of augmentations
                augmentation_methods = [
                    self._apply_noise,
                    self._apply_rotation,
                    self._apply_scaling,
                    self._apply_translation,
                    self._apply_finger_variation
                ]
                
                # Randomly select 2-4 augmentation methods
                num_methods = random.randint(2, 4)
                selected_methods = random.sample(augmentation_methods, num_methods)
                
                # Apply the selected augmentations in sequence
                for method in selected_methods:
                    augmented_landmarks = method(augmented_landmarks)
                
                # Create a new augmented sample
                augmented_sample = {
                    "landmarks": augmented_landmarks,
                    "label": label,
                    "is_dual_hand": is_dual_hand,
                    "augmented": True,
                    "augmentation_methods": [m.__name__ for m in selected_methods]
                }
                
                augmented_samples.append(augmented_sample)
                
            return augmented_samples
            
        except Exception as e:
            logging.error(f"Error augmenting {json_path}: {str(e)}")
            return []
    
    def augment_directory(self, input_subdir, output_subdir):
        """
        Augment all landmark files in a directory and save to the output directory.
        
        Args:
            input_subdir: Input directory containing landmark JSON files
            output_subdir: Output directory to save augmented files
        
        Returns:
            Tuple of (processed_files, augmented_files_created)
        """
        os.makedirs(output_subdir, exist_ok=True)
        
        processed_files = 0
        augmented_files_created = 0
        
        # Get all JSON files in the directory
        json_files = []
        for root, _, files in os.walk(input_subdir):
            for file in files:
                if file.endswith('.json'):
                    json_files.append(os.path.join(root, file))
        
        for json_path in tqdm(json_files, desc=f"Augmenting {os.path.basename(input_subdir)}"):
            # Determine the relative path structure
            rel_path = os.path.relpath(json_path, input_subdir)
            output_rel_dir = os.path.dirname(rel_path)
            
            # Create corresponding output directory
            if output_rel_dir:
                full_output_dir = os.path.join(output_subdir, output_rel_dir)
                os.makedirs(full_output_dir, exist_ok=True)
            else:
                full_output_dir = output_subdir
            
            # Augment the file
            augmented_samples = self.augment_file(json_path)
            
            # Save augmented samples
            for i, sample in enumerate(augmented_samples):
                filename = os.path.splitext(os.path.basename(json_path))[0]
                output_path = os.path.join(full_output_dir, f"{filename}_aug_{i+1}.json")
                
                try:
                    with open(output_path, 'w') as f:
                        json.dump(sample, f, indent=4)
                    augmented_files_created += 1
                except Exception as e:
                    logging.error(f"Error saving augmented file {output_path}: {str(e)}")
            
            processed_files += 1
            
            # Log progress periodically
            if processed_files % 100 == 0:
                logging.info(f"Processed {processed_files}/{len(json_files)} files")
        
        return processed_files, augmented_files_created
    
    def augment_dataset(self):
        """
        Augment all landmark files in the input directory and save to the output directory.
        Works with both flat directory structure and nested directories.
        """
        total_processed = 0
        total_augmented = 0
        
        # Check if input directory contains subdirectories with json files
        has_subdirs_with_json = False
        for item in os.listdir(self.input_dir):
            subdir_path = os.path.join(self.input_dir, item)
            if os.path.isdir(subdir_path):
                if any(f.endswith('.json') for f in os.listdir(subdir_path)):
                    has_subdirs_with_json = True
                    break
        
        # If input dir has subdirectories with JSON files, process each separately
        if has_subdirs_with_json:
            logging.info(f"Found subdirectories containing JSON files. Processing each separately.")
            for subdir in os.listdir(self.input_dir):
                subdir_path = os.path.join(self.input_dir, subdir)
                if not os.path.isdir(subdir_path):
                    continue
                    
                # Check if there are any JSON files in this subdir
                json_files = [f for f in os.listdir(subdir_path) if f.endswith('.json')]
                if not json_files:
                    continue
                
                # Create corresponding output subfolder
                output_subdir = os.path.join(self.output_dir, subdir)
                
                # Augment this subdirectory
                processed, augmented = self.augment_directory(subdir_path, output_subdir)
                total_processed += processed
                total_augmented += augmented
        else:
            # Process flat directory structure
            logging.info(f"Processing flat directory structure.")
            processed, augmented = self.augment_directory(self.input_dir, self.output_dir)
            total_processed += processed
            total_augmented += augmented
        
        logging.info(f"Augmentation complete. Processed {total_processed} original files. Created {total_augmented} augmented files.")


def combine_datasets(original_dir, augmented_dir, output_dir, preserve_structure=True):
    """
    Combine original and augmented datasets into a single directory structure.
    
    Args:
        original_dir: Directory containing original landmark data
        augmented_dir: Directory containing augmented landmark data
        output_dir: Directory to save the combined dataset
        preserve_structure: Whether to preserve the directory structure
    """
    logging.info(f"Combining datasets from {original_dir} and {augmented_dir} into {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Helper function to copy files while preserving directory structure
    def copy_files_with_structure(src_dir, dst_dir):
        file_count = 0
        
        for root, _, files in os.walk(src_dir):
            # Calculate relative path
            rel_path = os.path.relpath(root, src_dir)
            
            # Create corresponding output directory
            if rel_path != '.':
                target_dir = os.path.join(dst_dir, rel_path)
                os.makedirs(target_dir, exist_ok=True)
            else:
                target_dir = dst_dir
            
            # Copy JSON files
            for file in files:
                if file.endswith('.json'):
                    src_file = os.path.join(root, file)
                    dst_file = os.path.join(target_dir, file)
                    
                    try:
                        with open(src_file, 'r') as src:
                            data = json.load(src)
                        
                        with open(dst_file, 'w') as dst:
                            json.dump(data, dst, indent=4)
                        file_count += 1
                    except Exception as e:
                        logging.error(f"Error copying file {src_file}: {str(e)}")
        
        return file_count
    
    # Copy original data
    orig_count = copy_files_with_structure(original_dir, output_dir)
    logging.info(f"Copied {orig_count} original files")
    
    # Copy augmented data
    aug_count = copy_files_with_structure(augmented_dir, output_dir)
    logging.info(f"Copied {aug_count} augmented files")
    
    logging.info(f"Dataset combination complete. Total files: {orig_count + aug_count}")


if __name__ == "__main__":
    # Define directories
    landmarks_dir = "landmarks"  # Where our landmark files are stored
    augmented_dir = "aug/augmented_landmarks"  # Where augmented landmarks will be saved
    combined_dir = "aug/combined_landmarks"  # Where combined dataset will be saved
    
    # Create augmenter
    augmenter = LandmarkAugmenter(
        input_dir=landmarks_dir,
        output_dir=augmented_dir,
        augmentations_per_sample=5  # Generate 5 augmented samples per original sample
    )
    
    # Run augmentation
    augmenter.augment_dataset()
    
    # Combine original and augmented datasets
    combine_datasets(
        original_dir=landmarks_dir,
        augmented_dir=augmented_dir,
        output_dir=combined_dir
    )
    
    logging.info("Data augmentation process completed successfully.")