import os
import sys
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import cv2
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    DATASET_PATH, SVM_MODEL_PATH, SCALER_MODEL_PATH,
    SVM_KERNEL, SVM_C, SVM_GAMMA, SVM_PROBABILITY, TEST_SIZE, RANDOM_STATE,
    CLASSES, AUGMENTATION_ENABLED, AUGMENTATION_FACTOR, ROTATION_RANGE,
    BRIGHTNESS_RANGE, NOISE_FACTOR, ensure_models_dir
)
from src.utils import MediaPipeHandExtractor, augment_image, plot_confusion_matrix, print_classification_report


def load_dataset_with_augmentation():
    """Load dataset and apply augmentation"""
    print("="*60)
    print("LOADING DATASET")
    print("="*60)
    print(f"Dataset path: {DATASET_PATH}")
    print(f"Augmentation: {'Enabled' if AUGMENTATION_ENABLED else 'Disabled'}")
    if AUGMENTATION_ENABLED:
        print(f"Augmentation factor: {AUGMENTATION_FACTOR}x\n")
    
    extractor = MediaPipeHandExtractor()
    
    X = []
    y = []
    failed_images = 0
    
    for class_idx, class_name in enumerate(CLASSES):
        class_path = os.path.join(DATASET_PATH, class_name)
        if not os.path.exists(class_path):
            print(f"⚠️  Class {class_name} not found")
            continue
        
        image_files = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.png'))]
        
        print(f"Processing class {class_name}: {len(image_files)} images")
        
        for img_file in tqdm(image_files, desc=f"  {class_name}"):
            img_path = os.path.join(class_path, img_file)
            image = cv2.imread(img_path)
            
            if image is None:
                failed_images += 1
                continue
            
            # Extract original features
            features = extractor.extract_normalized_landmarks(image)
            
            if features is not None:
                X.append(features)
                y.append(class_idx)
                
                # Data Augmentation
                if AUGMENTATION_ENABLED:
                    for _ in range(AUGMENTATION_FACTOR - 1):
                        aug_image = augment_image(
                            image,
                            rotation_range=ROTATION_RANGE,
                            brightness_range=BRIGHTNESS_RANGE,
                            noise_factor=NOISE_FACTOR
                        )
                        aug_features = extractor.extract_normalized_landmarks(aug_image)
                        
                        if aug_features is not None:
                            X.append(aug_features)
                            y.append(class_idx)
            else:
                failed_images += 1
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"\n✅ Dataset loaded!")
    print(f"   Total samples: {len(X)}")
    print(f"   Feature shape: {X.shape}")
    print(f"   Failed images: {failed_images}")
    print(f"   Classes: {len(CLASSES)}\n")
    
    return X, y


def train_model(X_train, y_train):
    """Train SVM classifier"""
    print("="*60)
    print("TRAINING SVM")
    print("="*60)
    print(f"Kernel: {SVM_KERNEL}")
    print(f"C: {SVM_C}")
    print(f"Gamma: {SVM_GAMMA}\n")
    
    svm = SVC(
        kernel=SVM_KERNEL,
        C=SVM_C,
        gamma=SVM_GAMMA,
        probability=SVM_PROBABILITY,
        random_state=RANDOM_STATE,
        verbose=True
    )
    
    svm.fit(X_train, y_train)
    
    print(f"\n✅ SVM trained!")
    print(f"   Support vectors: {svm.n_support_.sum()}")
    print(f"   Classes: {len(svm.classes_)}\n")
    
    return svm


def evaluate_model(svm, X_train, y_train, X_test, y_test):
    """Evaluate model and check for overfitting"""
    print("="*60)
    print("EVALUATION")
    print("="*60)
    
    y_train_pred = svm.predict(X_train)
    y_test_pred = svm.predict(X_test)
    
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    diff = abs(train_acc - test_acc)
    
    print(f"\n📊 Accuracy:")
    print(f"   Training: {train_acc*100:.2f}%")
    print(f"   Testing: {test_acc*100:.2f}%")
    print(f"   Difference: {diff*100:.2f}%\n")
    
    if diff < 0.05:
        print("   ✅ Good generalization!")
    elif diff < 0.10:
        print("   ⚠️  Slight overfitting")
    else:
        print("   ❌ Overfitting detected!")
        print("   💡 Try increasing AUGMENTATION_FACTOR or decreasing SVM_C")
    
    print()
    print_classification_report(y_test, y_test_pred, CLASSES)
    
    return y_test_pred


def save_models(svm, scaler):
    """Save trained models"""
    print("="*60)
    print("SAVING MODELS")
    print("="*60)
    
    ensure_models_dir()
    
    joblib.dump(svm, SVM_MODEL_PATH)
    print(f"  ✅ SVM: {SVM_MODEL_PATH}")
    
    joblib.dump(scaler, SCALER_MODEL_PATH)
    print(f"  ✅ Scaler: {SCALER_MODEL_PATH}")
    
    print("\n🎉 Training completed successfully!")


def main():
    """Main training pipeline"""
    print("\n" + "="*60)
    print("ASL RECOGNITION - MEDIAPIPE TRAINING")
    print("="*60 + "\n")
    
    # Load dataset
    X, y = load_dataset_with_augmentation()
    
    # Split dataset
    print("="*60)
    print("TRAIN/TEST SPLIT")
    print("="*60)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE,
        stratify=y
    )
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}\n")
    
    # Scale features
    print("="*60)
    print("FEATURE SCALING")
    print("="*60)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"Train mean: {X_train_scaled.mean():.4f}")
    print(f"Train std: {X_train_scaled.std():.4f}\n")
    
    # Train model
    svm = train_model(X_train_scaled, y_train)
    
    # Evaluate
    y_test_pred = evaluate_model(svm, X_train_scaled, y_train, X_test_scaled, y_test)
    
    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_test_pred, CLASSES, "ASL Recognition - MediaPipe")
    
    # Save models
    save_models(svm, scaler)


if __name__ == "__main__":
    main()
