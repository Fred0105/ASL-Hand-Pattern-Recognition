import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "SignAlphaSet", "SignAlphaSet")
MODELS_DIR = os.path.join(BASE_DIR, "models")

SVM_MODEL_PATH = os.path.join(MODELS_DIR, "svm_model.pkl")
KMEANS_MODEL_PATH = os.path.join(MODELS_DIR, "kmeans_model.pkl")
SCALER_MODEL_PATH = os.path.join(MODELS_DIR, "scaler.pkl")

# Model parameters - MediaPipe Hand Landmarks
USE_MEDIAPIPE = True
MEDIAPIPE_CONFIDENCE = 0.3  # Lowered from 0.5 for easier hand detection
HAND_LANDMARKS_COUNT = 21
FEATURE_VECTOR_SIZE = 63  # 21 landmarks * 3 (x, y, z)

# Data Augmentation (prevent overfitting)
AUGMENTATION_ENABLED = False
ROTATION_RANGE = 15
BRIGHTNESS_RANGE = 0.2
NOISE_FACTOR = 0.02
AUGMENTATION_FACTOR = 2

# SVM parameters
SVM_KERNEL = "rbf"  # Changed to RBF for better generalization
SVM_C = 1.0
SVM_GAMMA = "scale"
SVM_PROBABILITY = True

# Training parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Inference parameters (optimized for better responsiveness)
CONFIDENCE_THRESHOLD = 0.65  # Lowered from 0.80 for easier detection
HISTORY_SIZE = 20            # Reduced from 30 for faster response
MIN_STABLE_FRAMES = 10       # Reduced from 15 (hold gesture ~0.5 seconds)
MIN_FREQUENCY = 7            # Reduced from 12 (70% consistency in window)
ROI_SIZE = 150
CAMERA_INDEX = 0

# Preprocessing
LOWER_SKIN = [0, 20, 70]
UPPER_SKIN = [20, 255, 255]
KERNEL_SIZE = (5, 5)
DILATE_ITERATIONS = 1
GAUSSIAN_BLUR_SIZE = (5, 5)
GAUSSIAN_BLUR_SIGMA = 50
CANNY_THRESHOLD1 = 50
CANNY_THRESHOLD2 = 150

# Classes
CLASSES = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
NUM_CLASSES = len(CLASSES)

# Visualization
CONFUSION_MATRIX_FIGSIZE = (14, 12)
CONFUSION_MATRIX_CMAP = "Blues"
FONT = 1
FONT_SCALE = 1
FONT_THICKNESS = 2
FONT_COLOR_GREEN = (0, 255, 0)
FONT_COLOR_YELLOW = (0, 255, 255)
FONT_COLOR_WHITE = (255, 255, 255)
FONT_COLOR_RED = (255, 0, 0)
FONT_COLOR_MAGENTA = (255, 0, 255)

def ensure_models_dir():
    os.makedirs(MODELS_DIR, exist_ok=True)
