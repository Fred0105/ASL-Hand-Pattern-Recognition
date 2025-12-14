import cv2
import numpy as np
import os
import sys
import mediapipe as mp
from typing import Optional, Tuple

# Ensure config can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from config import MEDIAPIPE_CONFIDENCE, HAND_LANDMARKS_COUNT


class MediaPipeHandExtractor:
    """Extract hand landmarks using MediaPipe"""
    
    def __init__(self, confidence=MEDIAPIPE_CONFIDENCE):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=confidence,
            min_tracking_confidence=confidence
        )
        
    def extract_landmarks(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Extract hand landmarks from image"""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        
        if not results.multi_hand_landmarks:
            return None
            
        hand_landmarks = results.multi_hand_landmarks[0]
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
            
        return np.array(landmarks, dtype=np.float32)
    
    def extract_normalized_landmarks(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Extract and normalize hand landmarks (translation & scale invariant)"""
        landmarks = self.extract_landmarks(image)
        
        if landmarks is None:
            return None
            
        landmarks = landmarks.reshape(21, 3)
        wrist = landmarks[0]
        landmarks = landmarks - wrist
        
        max_dist = np.max(np.linalg.norm(landmarks, axis=1))
        if max_dist > 0:
            landmarks = landmarks / max_dist
            
        return landmarks.flatten()
    
    
    def visualize_landmarks(self, image: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Draw hand landmarks on image in blue color"""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        
        annotated_image = image.copy()
        success = False
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw connections (lines) in blue
                connections = self.mp_hands.HAND_CONNECTIONS
                for connection in connections:
                    start_idx = connection[0]
                    end_idx = connection[1]
                    
                    start = hand_landmarks.landmark[start_idx]
                    end = hand_landmarks.landmark[end_idx]
                    
                    h, w, _ = annotated_image.shape
                    start_point = (int(start.x * w), int(start.y * h))
                    end_point = (int(end.x * w), int(end.y * h))
                    
                    # Blue lines (BGR format: Blue=255, Green=0, Red=0)
                    cv2.line(annotated_image, start_point, end_point, (255, 0, 0), 2)
                
                # Draw landmarks (points) in blue
                for landmark in hand_landmarks.landmark:
                    h, w, _ = annotated_image.shape
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    # Blue circles
                    cv2.circle(annotated_image, (cx, cy), 5, (255, 0, 0), -1)
            
            success = True
            
        return annotated_image, success
    
    def __del__(self):
        if hasattr(self, 'hands'):
            self.hands.close()


def augment_image(image: np.ndarray, 
                  rotation_range: int = 15,
                  brightness_range: float = 0.2,
                  noise_factor: float = 0.02) -> np.ndarray:
    """Apply data augmentation to prevent overfitting"""
    h, w = image.shape[:2]
    
    # Random rotation
    angle = np.random.uniform(-rotation_range, rotation_range)
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    image = cv2.warpAffine(image, M, (w, h))
    
    # Random brightness
    brightness = np.random.uniform(1 - brightness_range, 1 + brightness_range)
    image = np.clip(image * brightness, 0, 255).astype(np.uint8)
    
    # Add Gaussian noise
    if noise_factor > 0:
        noise = np.random.randn(*image.shape) * noise_factor * 255
        image = np.clip(image + noise, 0, 255).astype(np.uint8)
    
    return image

