import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os
import sys

# Ensure config can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from config import (
    CONFUSION_MATRIX_FIGSIZE, CONFUSION_MATRIX_CMAP,
    FONT, FONT_SCALE, FONT_THICKNESS,
    FONT_COLOR_GREEN, FONT_COLOR_YELLOW, FONT_COLOR_WHITE,
    FONT_COLOR_RED, FONT_COLOR_MAGENTA
)


def plot_confusion_matrix(y_true, y_pred, classes, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=CONFUSION_MATRIX_FIGSIZE)
    
    # Create heatmap with annotations (numbers)
    sns.heatmap(cm, annot=True, fmt='d', cmap=CONFUSION_MATRIX_CMAP,
                xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': 'Count'},
                linewidths=0.5, linecolor='gray')
    
    plt.xlabel("Predicted", fontsize=12, fontweight='bold')
    plt.ylabel("True", fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()


def print_classification_report(y_true, y_pred, classes):
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(y_true, y_pred, target_names=classes))
    print("="*60 + "\n")


def draw_text(frame, text, position, color=FONT_COLOR_WHITE):
    cv2.putText(frame, text, position, FONT, FONT_SCALE, color, FONT_THICKNESS)
    return frame


def draw_roi_box(frame, x1, y1, x2, y2, color=FONT_COLOR_GREEN):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    return frame


def create_display(frame, current_letter, confidence, sentence="", 
                   final_letter="", fps=0, roi_coords=None):
    if roi_coords is not None:
        x1, y1, x2, y2 = roi_coords
        frame = draw_roi_box(frame, x1, y1, x2, y2)
    
    frame = draw_text(frame, f"Current: {current_letter}", (30, 40), FONT_COLOR_GREEN)
    
    if final_letter:
        frame = draw_text(frame, f"Final: {final_letter}", (30, 80), FONT_COLOR_YELLOW)
    
    frame = draw_text(frame, f"Sentence: {sentence}", (30, 120), FONT_COLOR_WHITE)
    
    conf_color = FONT_COLOR_GREEN if confidence > 0.8 else FONT_COLOR_RED
    frame = draw_text(frame, f"Confidence: {round(confidence, 2)}", (30, 160), conf_color)
    
    if fps > 0:
        frame = draw_text(frame, f"FPS: {fps}", (30, 200), FONT_COLOR_MAGENTA)
    
    return frame
