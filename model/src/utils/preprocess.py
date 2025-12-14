import cv2
import numpy as np
import os
import sys

# Ensure config can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from config import (
    LOWER_SKIN, UPPER_SKIN, KERNEL_SIZE, DILATE_ITERATIONS,
    GAUSSIAN_BLUR_SIZE, GAUSSIAN_BLUR_SIGMA,
    CANNY_THRESHOLD1, CANNY_THRESHOLD2
)


def extract_roi(frame, center_x, center_y, size):
    h, w = frame.shape[:2]
    x1 = max(0, center_x - size)
    y1 = max(0, center_y - size)
    x2 = min(w, center_x + size)
    y2 = min(h, center_y + size)
    roi = frame[y1:y2, x1:x2]
    return roi, x1, y1, x2, y2


def detect_skin(image):
    lower_skin = np.array(LOWER_SKIN)
    upper_skin = np.array(UPPER_SKIN)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    return mask


def enhance_mask(mask):
    kernel = np.ones(KERNEL_SIZE, np.uint8)
    enhanced = cv2.dilate(mask, kernel, iterations=DILATE_ITERATIONS)
    enhanced = cv2.GaussianBlur(enhanced, GAUSSIAN_BLUR_SIZE, GAUSSIAN_BLUR_SIGMA)
    return enhanced


def detect_edges(image):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    edges = cv2.Canny(gray, CANNY_THRESHOLD1, CANNY_THRESHOLD2)
    return edges


def preprocess_for_inference(roi, use_skin_detection=True, use_edge_detection=True):
    if roi.size == 0:
        return None
    
    if use_skin_detection:
        mask = detect_skin(roi)
        mask = enhance_mask(mask)
        hand = cv2.bitwise_and(roi, roi, mask=mask)
        gray = cv2.cvtColor(hand, cv2.COLOR_BGR2GRAY)
    else:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    if use_edge_detection:
        processed = detect_edges(gray)
    else:
        processed = gray
    
    return processed
