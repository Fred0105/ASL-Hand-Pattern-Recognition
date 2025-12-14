from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import joblib
import os
import sys
from collections import deque, Counter
import threading

# Add model directory to path
model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'model')
sys.path.insert(0, model_dir)

from src.config import (
    SVM_MODEL_PATH, SCALER_MODEL_PATH,
    CLASSES, CONFIDENCE_THRESHOLD, 
    HISTORY_SIZE, MIN_STABLE_FRAMES, MIN_FREQUENCY
)
from src.utils import MediaPipeHandExtractor

app = Flask(__name__)

# Global variables for ASL recognition with thread lock
lock = threading.Lock()
svm_model = None
scaler = None
extractor = None
prediction_history = deque(maxlen=HISTORY_SIZE)
sentence = ""
last_final_letter = ""
current_letter = ""
current_confidence = 0.0

def load_models():
    """Load trained models"""
    global svm_model, scaler, extractor
    
    if not os.path.exists(SVM_MODEL_PATH) or not os.path.exists(SCALER_MODEL_PATH):
        raise FileNotFoundError(
            "Models not found! Please train the model first:\n"
            "cd model && python -m src.training.train"
        )
    
    svm_model = joblib.load(SVM_MODEL_PATH)
    scaler = joblib.load(SCALER_MODEL_PATH)
    extractor = MediaPipeHandExtractor()
    print("✅ Models loaded successfully!")

def predict_sign(frame):
    """Predict ASL sign from frame"""
    global prediction_history, sentence, last_final_letter, current_letter, current_confidence
    
    try:
        # Extract features
        features = extractor.extract_normalized_landmarks(frame)
        
        if features is None:
            current_letter = ""
            current_confidence = 0.0
            return frame, None, 0.0, False
        
        # Predict
        features_scaled = scaler.transform([features])
        pred_idx = svm_model.predict(features_scaled)[0]
        confidence = np.max(svm_model.predict_proba(features_scaled)[0])
        predicted_letter = CLASSES[pred_idx]
        
        # Update global stats for API
        current_letter = predicted_letter
        current_confidence = confidence
        
        # Update sentence with stable predictions (thread-safe)
        with lock:
            if confidence > CONFIDENCE_THRESHOLD:
                prediction_history.append(predicted_letter)
            
            if len(prediction_history) >= MIN_STABLE_FRAMES:
                most_common = Counter(prediction_history).most_common(1)[0]
                final_letter = most_common[0]
                freq = most_common[1]
                
                if freq >= MIN_FREQUENCY and final_letter != last_final_letter:
                    sentence += final_letter
                    last_final_letter = final_letter
                    prediction_history.clear()
        
        # Draw landmarks
        annotated_frame = draw_landmarks(frame)
        
        return annotated_frame, predicted_letter, confidence, True
    
    except Exception as e:
        print(f"Prediction error: {e}")
        current_letter = ""
        current_confidence = 0.0
        return frame, None, 0.0, False

def draw_landmarks(frame):
    """Draw hand landmarks in blue color"""
    try:
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = extractor.hands.process(image_rgb)
        
        annotated_frame = frame.copy()
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw connections (lines) in blue
                connections = extractor.mp_hands.HAND_CONNECTIONS
                for connection in connections:
                    start_idx = connection[0]
                    end_idx = connection[1]
                    
                    start = hand_landmarks.landmark[start_idx]
                    end = hand_landmarks.landmark[end_idx]
                    
                    h, w, _ = annotated_frame.shape
                    start_point = (int(start.x * w), int(start.y * h))
                    end_point = (int(end.x * w), int(end.y * h))
                    
                    # Blue lines (BGR format)
                    cv2.line(annotated_frame, start_point, end_point, (255, 0, 0), 2)
                
                # Draw landmarks (points) in blue
                for landmark in hand_landmarks.landmark:
                    h, w, _ = annotated_frame.shape
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(annotated_frame, (cx, cy), 5, (255, 0, 0), -1)
        
        return annotated_frame
    except Exception as e:
        print(f"Landmark drawing error: {e}")
        return frame

def draw_ui(frame, predicted_letter, confidence, hand_detected):
    """Camera feed with ONLY landmarks - NO overlays, NO text"""
    # Return frame as-is - all info will be shown in the sidebar
    return frame

def generate_frames():
    """Generate video frames with ASL recognition"""
    camera = cv2.VideoCapture(0)
    # Reduced resolution for better performance
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_FPS, 30)  # Limit FPS
    
    if not camera.isOpened():
        print("❌ Error: Could not open camera")
        return
    
    try:
        while True:
            success, frame = camera.read()
            if not success:
                print("❌ Error: Could not read frame")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Predict sign
            annotated_frame, predicted_letter, confidence, hand_detected = predict_sign(frame)
            
            # Draw UI
            display_frame = draw_ui(annotated_frame, predicted_letter, confidence, hand_detected)
            
            # Encode frame with lower quality for faster streaming
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]  # 80% quality
            ret, buffer = cv2.imencode('.jpg', display_frame, encode_param)
            
            if not ret:
                continue
                
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    except Exception as e:
        print(f"Frame generation error: {e}")
    finally:
        camera.release()
        print("Camera released")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/camera')
def camera():
    return render_template('camera.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_stats')
def get_stats():
    """Get current detection stats including match percentage"""
    global current_letter, current_confidence
    
    with lock:
        current_sentence = sentence
        history_len = len(prediction_history)
    
    # Calculate match percentage (how close to adding letter)
    match_percentage = min(100, int((history_len / MIN_STABLE_FRAMES) * 100))
    
    return jsonify({
        'sentence': current_sentence,
        'letter': current_letter if current_letter else '-',
        'confidence': int(current_confidence * 100),
        'match': match_percentage
    })

@app.route('/get_sentence')
def get_sentence():
    """Get current sentence"""
    with lock:
        current_sentence = sentence
    return jsonify({'sentence': current_sentence})

@app.route('/reset_sentence')
def reset_sentence():
    """Reset sentence"""
    global sentence, last_final_letter, prediction_history
    with lock:
        sentence = ""
        last_final_letter = ""
        prediction_history.clear()
    return jsonify({'status': 'success'})

@app.route('/add_space')
def add_space():
    """Add space to sentence"""
    global sentence
    with lock:
        sentence += " "
        current_sentence = sentence
    return jsonify({'sentence': current_sentence})

@app.route('/delete_last')
def delete_last():
    """Delete last character"""
    global sentence
    with lock:
        sentence = sentence[:-1]
        current_sentence = sentence
    return jsonify({'sentence': current_sentence})

@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors"""
    return render_template('404.html'), 404

if __name__ == '__main__':
    try:
        print("🚀 Starting ASL Recognition Web App...")
        load_models()
        print("🌐 Server starting on http://localhost:5000")
        print("📝 Press Ctrl+C to stop")
        # Disable debug mode to prevent auto-reload lag
        app.run(debug=False, host='0.0.0.0', port=5001, threaded=True)
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("Please train the model first!")
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
