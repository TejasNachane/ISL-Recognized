import json
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import logging
import matplotlib.pyplot as plt
from datetime import datetime

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class HandGestureClassifier:
    """Class for training and evaluating a hand gesture classifier using landmark data."""
    
    def __init__(self, landmarks_dir):
        """
        Initialize the classifier.
        
        Args:
            landmarks_dir: Directory containing hand landmark data (combined original and augmented)
        """
        self.landmarks_dir = landmarks_dir
        self.label_encoder = LabelEncoder()
        self.model = None
        self.history = None
        
        # Constant indicating the expected number of landmarks per hand
        self.LANDMARKS_PER_HAND = 21
        self.COORDS_PER_LANDMARK = 3  # x, y, z
        
    def load_dataset(self):
        """
        Load landmarks and labels from JSON files.
        
        Returns:
            tuple: (features, labels) arrays if successful, (None, None) otherwise
        """
        logging.info(f"Loading dataset from {self.landmarks_dir}...")
        
        landmarks_list = []
        labels_list = []
        
        # Process directory structure
        for root, _, files in os.walk(self.landmarks_dir):
            for file in files:
                if not file.endswith('.json'):
                    continue
                    
                file_path = os.path.join(root, file)
                
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        
                    # Validate data structure
                    if 'landmarks' not in data or not data['landmarks']:
                        logging.warning(f"Skipping {file_path}: No landmarks found.")
                        continue
                        
                    # Get label (either from data or directory structure)
                    if 'label' in data:
                        label = data['label']
                    else:
                        # Use directory name as label if available
                        label = os.path.basename(root)
                        if label == os.path.basename(self.landmarks_dir):
                            # If in the root directory, use filename base
                            label = os.path.splitext(file)[0].split('_')[0]
                    
                    # Process landmarks (handling both hands)
                    hand_landmarks = data['landmarks']
                    
                    # Flatten the landmarks into a feature vector
                    flattened = self._flatten_landmarks(hand_landmarks)
                    
                    landmarks_list.append(flattened)
                    labels_list.append(label)
                    
                except Exception as e:
                    logging.error(f"Error processing {file_path}: {str(e)}")
        
        if not landmarks_list:
            logging.error("No valid landmark data found.")
            return None, None
            
        logging.info(f"Loaded {len(landmarks_list)} samples with {len(set(labels_list))} unique classes.")
        
        return np.array(landmarks_list, dtype=np.float32), np.array(labels_list)
    
    def _flatten_landmarks(self, hand_landmarks):
        """
        Convert hand landmarks to a flat feature vector.
        
        Args:
            hand_landmarks: List of hand landmarks from the JSON file
            
        Returns:
            numpy.ndarray: Flattened feature vector
        """
        # Ensure we have data for exactly 2 hands (may include zero-padding)
        if len(hand_landmarks) > 2:
            hand_landmarks = hand_landmarks[:2]  # Use only first two hands
        elif len(hand_landmarks) < 2:
            # Add zero padding for missing hand
            zero_hand = [[0, 0, 0] for _ in range(self.LANDMARKS_PER_HAND)]
            hand_landmarks.append(zero_hand)
        
        # Flatten the structure
        feature_vector = []
        for hand in hand_landmarks:
            for landmark in hand:
                feature_vector.extend(landmark)  # Add x, y, z
        
        return np.array(feature_vector)
    
    def preprocess_data(self, features, labels):
        """
        Preprocess the data for training.
        
        Args:
            features: Landmark features
            labels: Class labels
            
        Returns:
            tuple: Preprocessed data splits and number of classes
        """
        # Encode labels as integers
        encoded_labels = self.label_encoder.fit_transform(labels)
        num_classes = len(self.label_encoder.classes_)
        
        logging.info(f"Classes: {self.label_encoder.classes_}")
        
        # Save label encoder mapping
        label_mapping = {str(i): label for i, label in enumerate(self.label_encoder.classes_)}
        with open('label_mapping.json', 'w') as f:
            json.dump(label_mapping, f, indent=4)
        logging.info("Saved label mapping to label_mapping.json")
        
        # Split into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            features, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels
        )
        
        logging.info(f"Training set: {X_train.shape[0]} samples")
        logging.info(f"Validation set: {X_val.shape[0]} samples")
        
        return X_train, X_val, y_train, y_val, num_classes
    
    def build_model(self, input_shape, num_classes):
        """
        Build and compile the model.
        
        Args:
            input_shape: Shape of input features
            num_classes: Number of output classes
            
        Returns:
            tensorflow.keras.Model: Compiled model
        """
        model = models.Sequential([
            # Input layer
            layers.Input(shape=input_shape),
            
            # First block
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Second block
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Third block
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            # Output layer
            layers.Dense(num_classes, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Print model summary
        model.summary()
        
        self.model = model
        return model
    
    def train(self, X_train, X_val, y_train, y_val, batch_size=32, epochs=100):
        """
        Train the model.
        
        Args:
            X_train: Training features
            X_val: Validation features
            y_train: Training labels
            y_val: Validation labels
            batch_size: Batch size
            epochs: Maximum number of epochs
            
        Returns:
            tensorflow.keras.callbacks.History: Training history
        """
        # Create logs directory
        log_dir = os.path.join('logs', datetime.now().strftime("%Y%m%d-%H%M%S"))
        os.makedirs(log_dir, exist_ok=True)
        
        # Define callbacks
        callbacks = [
            # Early stopping to prevent overfitting
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', 
                patience=15, 
                restore_best_weights=True
            ),
            # Reduce learning rate when plateauing
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', 
                factor=0.5, 
                patience=5, 
                min_lr=1e-6
            ),
            # TensorBoard logging
            tf.keras.callbacks.TensorBoard(log_dir=log_dir)
        ]
        
        # Train the model
        logging.info("Starting model training...")
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        self.history = history
        return history
    
    def evaluate(self, X_val, y_val):
        """
        Evaluate the model on validation data.
        
        Args:
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            tuple: (loss, accuracy)
        """
        if self.model is None:
            logging.error("Model not trained yet.")
            return None
            
        logging.info("Evaluating model...")
        loss, accuracy = self.model.evaluate(X_val, y_val, verbose=1)
        logging.info(f"Validation Loss: {loss:.4f}")
        logging.info(f"Validation Accuracy: {accuracy:.4f}")
        
        return loss, accuracy
    
    def save_model(self, file_path):
        """
        Save the trained model to disk.
        
        Args:
            file_path: Path to save model
        """
        if self.model is None:
            logging.error("No model to save.")
            return
            
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Save the model
            self.model.save(file_path)
            logging.info(f"Model saved to {file_path}")
            
            # Save model architecture as JSON
            model_json = self.model.to_json()
            with open(f"{os.path.splitext(file_path)[0]}.json", "w") as json_file:
                json_file.write(model_json)
            logging.info(f"Model architecture saved to {os.path.splitext(file_path)[0]}.json")
        except Exception as e:
            logging.error(f"Error saving model: {str(e)}")
    
    def plot_training_history(self, save_path=None):
        """
        Plot training history.
        
        Args:
            save_path: Path to save the plot (optional)
        """
        if self.history is None:
            logging.error("No training history available.")
            return
            
        plt.figure(figsize=(12, 5))
        
        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['accuracy'], label='Training Accuracy')
        plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logging.info(f"Training history plot saved to {save_path}")
        else:
            plt.show()


def main():
    """Main function to run the training pipeline."""
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Define paths
    landmarks_dir = "aug/augmented_landmarks"  # Path to the combined dataset from augmentation
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "hand_gesture_classifier.h5")
    
    # Initialize classifier
    classifier = HandGestureClassifier(landmarks_dir)
    
    try:
        # Load and preprocess data
        features, labels = classifier.load_dataset()
        if features is None or labels is None:
            logging.error("Failed to load dataset.")
            return
            
        X_train, X_val, y_train, y_val, num_classes = classifier.preprocess_data(features, labels)
        
        # Build and train model
        classifier.build_model(input_shape=(features.shape[1],), num_classes=num_classes)
        classifier.train(X_train, X_val, y_train, y_val, batch_size=32, epochs=100)
        
        # Evaluate model
        classifier.evaluate(X_val, y_val)
        
        # Save model
        classifier.save_model(model_path)
        
        # Plot and save training history
        classifier.plot_training_history(save_path=os.path.join(model_dir, "training_history.png"))
        
        logging.info("Training pipeline completed successfully!")
        
    except Exception as e:
        logging.error(f"Error in training pipeline: {str(e)}")


if __name__ == "__main__":
    main()