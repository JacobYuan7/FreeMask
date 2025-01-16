import numpy as np
from PIL import Image
import cv2
import argparse

def calculate_miou(ground_truth, prediction, n_classes):
    confusion_matrix = np.zeros((n_classes, n_classes))
    for gt, pred in zip(ground_truth.flatten(), prediction.flatten()):
        confusion_matrix[gt][pred] += 1

    ious = np.diag(confusion_matrix) / (
        np.sum(confusion_matrix, axis=0) + np.sum(confusion_matrix, axis=1) - np.diag(confusion_matrix)
    )
    ious = ious[~np.isnan(ious)]
    miou = np.mean(ious)

    return miou


def load_image_as_binary_array(image_path, threshold=10):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    binary_image = np.array(binary_image)
    return binary_image

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate mIoU between ground truth and prediction.')
    parser.add_argument('ground_truth_path', type=str, help='Path to the ground truth image')
    parser.add_argument('prediction_path', type=str, help='Path to the prediction image')

    args = parser.parse_args()

    ground_truth = load_image_as_binary_array(args.ground_truth_path)
    prediction = load_image_as_binary_array(args.prediction_path)

    ground_truth = ground_truth // 255
    prediction = prediction // 255

    miou = calculate_miou(ground_truth, prediction, 2)
    print('MIoU: ', miou)
