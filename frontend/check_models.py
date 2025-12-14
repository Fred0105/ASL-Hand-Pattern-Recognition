"""
Quick script to check if models are trained and ready
"""
import os
import sys

# Add model directory to path
model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'model')
sys.path.insert(0, model_dir)

from src.config import SVM_MODEL_PATH, SCALER_MODEL_PATH, MODELS_DIR

def check_models():
    print("🔍 Checking for trained models...\n")
    
    print(f"Models directory: {MODELS_DIR}")
    print(f"Expected SVM model: {SVM_MODEL_PATH}")
    print(f"Expected Scaler: {SCALER_MODEL_PATH}\n")
    
    svm_exists = os.path.exists(SVM_MODEL_PATH)
    scaler_exists = os.path.exists(SCALER_MODEL_PATH)
    
    if svm_exists:
        print("✅ SVM model found!")
        print(f"   Size: {os.path.getsize(SVM_MODEL_PATH) / (1024*1024):.2f} MB")
    else:
        print("❌ SVM model NOT found!")
    
    if scaler_exists:
        print("✅ Scaler found!")
        print(f"   Size: {os.path.getsize(SCALER_MODEL_PATH) / 1024:.2f} KB")
    else:
        print("❌ Scaler NOT found!")
    
    print()
    
    if svm_exists and scaler_exists:
        print("🎉 All models are ready! You can run the web app:")
        print("   python app.py")
    else:
        print("⚠️  Models not found! Please train the model first:")
        print("   cd ../model")
        print("   python -m src.training.train")

if __name__ == "__main__":
    check_models()
