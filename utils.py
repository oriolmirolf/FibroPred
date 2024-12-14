# utils.py

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
import numpy as np

def plot_confusion_matrix(y_true, y_prob, threshold=0.5, set_type='Train'):
    """
    Plots a confusion matrix using Seaborn's heatmap based on predicted probabilities and a threshold.

    Parameters:
    - y_true (array-like): True binary labels.
    - y_prob (array-like): Predicted probabilities for the positive class.
    - threshold (float): Threshold to convert probabilities to binary predictions.
    - title (str): Title of the plot.
    """
    # Validate inputs
    if len(y_true) != len(y_prob):
        raise ValueError("y_true and y_prob must have the same length.")
    
    # Compute predicted labels based on the threshold
    y_pred = (np.array(y_prob) >= threshold).astype(int)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Set up the plot
    sns.set(font_scale=1.2)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f"Confusion Matrix - Progressive Disease ({set_type} Set)")
    plt.tight_layout()
    plt.show()

def plot_roc_curve(y_true, y_prob, set_type='Train'):
    """
    Plots an ROC curve and displays the AUC score.

    Parameters:
    - y_true (array-like): True binary labels.
    - y_prob (array-like): Predicted probabilities for the positive class.
    - title (str): Title of the plot.
    """
    # Validate inputs
    if len(y_true) != len(y_prob):
        raise ValueError("y_true and y_prob must have the same length.")
    
    # Compute ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auc_score = roc_auc_score(y_true, y_prob)
    
    # Set up the plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.2f})', color='blue')
    plt.plot([0, 1], [0, 1], 'r--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f"ROC Curve - Progressive Disease ({set_type} Set)")
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()

def visualize_all(y_true, y_prob, threshold=0.5, set_type='Train'):
    """
    Executes both plot_confusion_matrix and plot_roc_curve functions.

    Parameters:
    - y_true (array-like): True binary labels.
    - y_prob (array-like): Predicted probabilities for the positive class.
    - threshold (float): Threshold to convert probabilities to binary predictions.
    - cm_title (str): Title for the confusion matrix plot.
    - roc_title (str): Title for the ROC curve plot.
    """
    plot_confusion_matrix(y_true, y_prob, threshold=threshold, set_type=set_type)
    plot_roc_curve(y_true, y_prob, set_type)
