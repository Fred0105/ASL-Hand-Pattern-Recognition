import os
import sys
import cv2
import numpy as np
import joblib
import time
from collections import deque, Counter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    SVM_MODEL_PATH, SCALER_MODEL_PATH,
    CLASSES, CAMERA_INDEX,
    CONFIDENCE_THRESHOLD, HISTORY_SIZE, MIN_STABLE_FRAMES, MIN_FREQUENCY
)
from src.utils import MediaPipeHandExtractor


class ASLRecognizer:
    def __init__(self):
        print("Loading models...")
        
        if not os.path.exists(SVM_MODEL_PATH) or not os.path.exists(SCALER_MODEL_PATH):
            raise FileNotFoundError(
                "Models not found! Please train the model first:\n"
                "python -m src.training.train"
            )
        
        self.svm = joblib.load(SVM_MODEL_PATH)
        self.scaler = joblib.load(SCALER_MODEL_PATH)
        self.extractor = MediaPipeHandExtractor()
        
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        if not self.cap.isOpened():
            raise RuntimeError("Camera not found!")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        self.history = deque(maxlen=HISTORY_SIZE)
        self.sentence = ""
        self.last_final_letter = ""
        self.prev_time = 0
        
        print("✅ Ready! Press 'q' to quit\n")
    
    def predict(self, image):
        """Predict ASL sign from image"""
        features = self.extractor.extract_normalized_landmarks(image)
        
        if features is None:
            return None, 0.0, False
        
        features_scaled = self.scaler.transform([features])
        pred_idx = self.svm.predict(features_scaled)[0]
        confidence = np.max(self.svm.predict_proba(features_scaled)[0])
        
        return CLASSES[pred_idx], confidence, True
    
    def update_sentence(self, current_letter, confidence):
        """Update sentence based on stable predictions"""
        if confidence > CONFIDENCE_THRESHOLD:
            self.history.append(current_letter)
        
        if len(self.history) >= MIN_STABLE_FRAMES:
            most_common = Counter(self.history).most_common(1)[0]
            final_letter = most_common[0]
            freq = most_common[1]
            
            if freq >= MIN_FREQUENCY and final_letter != self.last_final_letter:
                self.sentence += final_letter
                self.last_final_letter = final_letter
                self.history.clear()
                return final_letter
        return ""
    
    def draw_blue_landmarks(self, frame):
        """Draw hand landmarks in blue color only"""
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.extractor.hands.process(image_rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw connections (lines) in blue
                connections = self.extractor.mp_hands.HAND_CONNECTIONS
                for connection in connections:
                    start_idx = connection[0]
                    end_idx = connection[1]
                    
                    start = hand_landmarks.landmark[start_idx]
                    end = hand_landmarks.landmark[end_idx]
                    
                    h, w, _ = frame.shape
                    start_point = (int(start.x * w), int(start.y * h))
                    end_point = (int(end.x * w), int(end.y * h))
                    
                    # Blue lines
                    cv2.line(frame, start_point, end_point, (255, 0, 0), 2)
                
                # Draw landmarks (points) in blue
                for landmark in hand_landmarks.landmark:
                    h, w, _ = frame.shape
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    # Blue circles
                    cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
        
        return frame
    
    def draw_clean_display(self, frame, current_letter, confidence, hand_detected, fps):
        """Draw clean and simple UI"""
        h, w = frame.shape[:2]
        
        # Draw blue landmarks
        frame = self.draw_blue_landmarks(frame)
        
        # Prediction text (top left)
        if hand_detected:
            text = f"Prediction: {current_letter} ({confidence*100:.1f}%)"
            color = (0, 255, 0) if confidence >= CONFIDENCE_THRESHOLD else (0, 165, 255)
        else:
            text = "No hand detected"
            color = (0, 0, 255)
        
        cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        
        # Sentence (bottom)
        sentence_text = f"Sentence: {self.sentence}"
        cv2.putText(frame, sentence_text, (20, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Controls (bottom)
        controls = "Q: Quit | SPACE: Space | BACKSPACE: Delete | R: Reset"
        cv2.putText(frame, controls, (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # FPS (top right)
        cv2.putText(frame, f"FPS: {fps}", (w - 120, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def run(self):
        """Run real-time recognition"""
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                frame = cv2.flip(frame, 1)
                
                # Calculate FPS
                curr_time = time.time()
                fps = int(1 / (curr_time - self.prev_time)) if self.prev_time != 0 else 0
                self.prev_time = curr_time
                
                # Predict
                current_letter, confidence, hand_detected = self.predict(frame)
                
                # Update sentence
                if hand_detected:
                    self.update_sentence(current_letter, confidence)
                
                # Draw clean display
                display_frame = self.draw_clean_display(frame, current_letter or "", confidence, hand_detected, fps)
                
                # Show frame
                cv2.imshow("ASL Recognition", display_frame)
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == 32:
                    self.sentence += " "
                elif key == 8:
                    self.sentence = self.sentence[:-1]
                elif key == ord('r'):
                    self.history.clear()
                    self.last_final_letter = ""
        
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            print("\n✅ Stopped")


if __name__ == "__main__":
    try:
        recognizer = ASLRecognizer()
        recognizer.run()
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
