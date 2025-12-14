import os
import sys
import cv2
import joblib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import SVM_MODEL_PATH, SCALER_MODEL_PATH, CLASSES, DATASET_PATH
from src.utils import MediaPipeHandExtractor


def recognize_image(image_path):
    """Recognize ASL sign from static image using MediaPipe keypoints"""
    
    # Load models
    if not os.path.exists(SVM_MODEL_PATH) or not os.path.exists(SCALER_MODEL_PATH):
        raise FileNotFoundError(
            "Models not found! Please train the model first:\n"
            "python -m src.training.train"
        )
    
    svm = joblib.load(SVM_MODEL_PATH)
    scaler = joblib.load(SCALER_MODEL_PATH)
    extractor = MediaPipeHandExtractor()
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Extract MediaPipe keypoints
    features = extractor.extract_normalized_landmarks(image)
    
    if features is None:
        print("❌ No hand detected in image!")
        return None, 0.0
    
    # Predict
    features_scaled = scaler.transform([features])
    pred_idx = svm.predict(features_scaled)[0]
    confidence = max(svm.predict_proba(features_scaled)[0])
    
    predicted_class = CLASSES[pred_idx]
    
    # Visualize
    annotated_image, _ = extractor.visualize_landmarks(image)
    
    # Draw prediction
    h, w = annotated_image.shape[:2]
    text = f"Prediction: {predicted_class} ({confidence*100:.1f}%)"
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    thickness = 3
    
    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    cv2.rectangle(annotated_image, (10, 10), (20 + text_w, 30 + text_h), (0, 0, 0), -1)
    cv2.putText(annotated_image, text, (15, 40), font, font_scale, (0, 255, 0), thickness)
    
    # Show result
    cv2.imshow("ASL Recognition - Static Image", annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return predicted_class, confidence


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="ASL Recognition - Static Image")
    parser.add_argument("image_path", nargs="?", help="Path to image file")
    parser.add_argument("--sample", action="store_true", help="Use sample image from dataset")
    
    args = parser.parse_args()
    
    # Determine image path
    if args.sample or not args.image_path:
        # Use sample from dataset
        sample_class = "A"
        sample_dir = os.path.join(DATASET_PATH, sample_class)
        if os.path.exists(sample_dir):
            files = [f for f in os.listdir(sample_dir) if f.endswith(('.jpg', '.png'))]
            if files:
                image_path = os.path.join(sample_dir, files[0])
                print(f"Using sample image: {image_path}")
            else:
                print("❌ No sample images found!")
                return
        else:
            print(f"❌ Sample directory not found: {sample_dir}")
            return
    else:
        image_path = args.image_path
    
    # Recognize
    try:
        predicted_class, confidence = recognize_image(image_path)
        
        if predicted_class:
            print(f"\n✅ Prediction: {predicted_class}")
            print(f"   Confidence: {confidence*100:.1f}%")
    
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()
