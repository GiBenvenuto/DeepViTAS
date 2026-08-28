# -------------------------------
# Numerical and Statistical Libraries
# -------------------------------
import numpy as np
import math
import statistics as s



# -------------------------------
# File and System Management
# -------------------------------
import os

# -------------------------------
# Image Processing
# -------------------------------
import cv2


# -------------------------------
# Evaluation Metrics
# -------------------------------
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    mean_squared_error,
    cohen_kappa_score,
    confusion_matrix,
    roc_auc_score,
)


# -------------------------------
# Similarity Metrics
# -------------------------------

def rmse(src, tgt):
  MSE = mean_squared_error(src, tgt)
  RMSE = math.sqrt(MSE)
  return RMSE

def dice(src, tgt):
  intersection = np.sum(src * tgt)
  denom = (np.sum(src) + np.sum(tgt))
  if denom > 0:
    score = (2. * intersection) / denom
  else:
    score = 1.0
  return score

def calculate_overall_accuracy(gt, pred):


    # TP, TN, FP, FN
    TP = np.sum((gt == 1) & (pred == 1))
    TN = np.sum((gt == 0) & (pred == 0))
    FP = np.sum((gt == 0) & (pred == 1))
    FN = np.sum((gt == 1) & (pred == 0))

    # Total
    total_pixels = TP + TN + FP + FN
    overall_accuracy = (TP + TN) / total_pixels

    return overall_accuracy

def calculate_iou (ground_truth, predicted):
    intersection = np.logical_and(ground_truth, predicted)
    union = np.logical_or(ground_truth, predicted)

    iou = np.sum(intersection) / np.sum(union)

    return iou


def load_bin(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    img = img/255

    return img

def precision(pred, gt):
    """
    Precision for binary segmentation.

    Args:
        pred (numpy array): Pred Mask (binary: 0 ou 1).
        gt (numpy array): Ground truth Mask (binary: 0 ou 1).

    Returns:
        float: Precision.
    """
    # Flatten arrays
    pred = pred.flatten()
    gt = gt.flatten()

    # Calcula VP e FP
    VP = np.sum((pred == 1) & (gt == 1))
    FP = np.sum((pred == 1) & (gt == 0))

    # Calcula precisão
    return VP / (VP + FP) if (VP + FP) > 0 else 0

def recall(pred, gt):
    """
    Recall for binary segmentation.

    Args:
        pred (numpy array): Pred Mask (binary: 0 ou 1).
        gt (numpy array): Ground truth Mask (binary: 0 ou 1).

    Returns:
        float: Recall.
    """
    # Flatten arrays
    pred = pred.flatten()
    gt = gt.flatten()

    # Calcula VP e FN
    VP = np.sum((pred == 1) & (gt == 1))
    FN = np.sum((pred == 0) & (gt == 1))

    # Calcula recall
    return VP / (VP + FN) if (VP + FN) > 0 else 0

def printresults(name, vetor):
  print(name)
  print(vetor)
  vetor = [0.0 if math.isnan(x) else x for x in vetor]
  print(name + " Mean: ", s.mean(vetor))
  print(name + " STD: ", s.stdev(vetor))
  media= s.mean(vetor)
  desvio_padrao= s.stdev(vetor)
  print(f"{media:.4f} / {desvio_padrao:.4f}")


def apply_metrics_altamira(path_result, path_gt):
  dice_all = []
  f1_all = []
  #rmse_all = []
  iou_all = []
  #cc_all = []
  #mre_all = []
  #kappa_all = []
  precision_all = []
  recall_all = []


  for i in range(1, 50):
    #print(path_result + str(i) + '.png')
    img = load_bin(path_result + str(i) + '.png')
    gt = load_bin(path_gt + str(i) + '.png')
    dice_all.append(dice(gt, img))
    f1_all.append(f1_score(gt.flatten(), img.flatten()))
    #rmse_all.append(rmse(gt, img))
    iou_all.append(calculate_iou(gt, img))
    #corr, _ = pearsonr(gt.flatten(), img.flatten())
    #cc_all.append(corr)
    #kappa_all.append(cohen_kappa_score(gt.flatten(), img.flatten()))
    precision_all.append(precision(img, gt))
    recall_all.append(recall(img, gt))

  printresults("DICE", dice_all)
  printresults("F1 Score", f1_all)
  #printresults("RMSE", rmse_all)
  printresults("IoU", iou_all)
  #printresults("CC", cc_all)
  #printresults("Kappa ", kappa_all)
  printresults("Precision", precision_all)
  printresults("Recall", recall_all)
  return dice_all, iou_all, precision_all, recall_all


def apply_metrics_amazon(path_result, path_gt):
  dice_all = []
  #f1_all = []
  #rmse_all = []
  iou_all = []
  #cc_all = []
  #mre_all = []
  oa_all = []
  kappa_all = []
  precision_all = []
  recall_all = []

  # List files and directories at the current level
  entries = os.listdir(path_gt)

  # Filter only files (if necessary)
  files = [f for f in entries if os.path.isfile(os.path.join(path_gt, f))]

  for i in files:
    #name_result = prefix + i.replace('.png', '.tif', 1) + '.png'
    #print(path_result + i)
    #img = load_bin(path_result + name_result)
    img = load_bin(path_result + i)
    gt = load_bin(path_gt + i)

    # Checks whether the gt image is completely white
    if np.all(gt == 1):
      print(f"Image {i} invalid. Next...")
      continue  # Next it
    #cv2_imshow(img*255)
    #cv2_imshow(gt*255)
    #print(path_gt + i)
    dice_all.append(dice(gt, img))
    #f1_all.append(f1_score(gt.flatten(), img.flatten()))
    oa_all.append(calculate_overall_accuracy(gt, img))
    #rmse_all.append(rmse(gt, img))
    iou_all.append(calculate_iou(gt, img))
    #corr, _ = pearsonr(gt.flatten(), img.flatten())
    #cc_all.append(corr)
    kappa_all.append(cohen_kappa_score(gt.flatten(), img.flatten()))
    precision_all.append(precision(img, gt))
    recall_all.append(recall(img, gt))

  printresults("DICE", dice_all)
  #printresults("F1 Score", f1_all)
  #printresults("RMSE", rmse_all)
  printresults("IoU", iou_all)
  printresults("OA", oa_all)
  #printresults("CC", cc_all)
  printresults("Kappa ", kappa_all)
  printresults("Precision", precision_all)
  printresults("Recall", recall_all)
  return dice_all, iou_all, precision_all, recall_all


def apply_metrics_altamira_512(path_result, path_gt):
  dice_all = []
  #f1_all = []
  #rmse_all = []
  iou_all = []
  #cc_all = []
  #mre_all = []
  kappa_all = []
  precision_all = []
  recall_all = []


  for i in range(10, 72):
    #print(path_result + str(i) + '.png')
    img = load_bin(path_result + str(i) + '.png')
    gt = load_bin(path_gt + 'Altamira_2020_' + str(i) + '/cm/cm.png')
    dice_all.append(dice(gt, img))
    #f1_all.append(f1_score(gt.flatten(), img.flatten()))
    #rmse_all.append(rmse(gt, img))
    iou_all.append(calculate_iou(gt, img))
    #corr, _ = pearsonr(gt.flatten(), img.flatten())
    #cc_all.append(corr)
    kappa_all.append(cohen_kappa_score(gt.flatten(), img.flatten()))
    precision_all.append(precision(img, gt))
    recall_all.append(recall(img, gt))

  printresults("DICE", dice_all)
  #printresults("F1 Score", f1_all)
  #printresults("RMSE", rmse_all)
  printresults("IoU", iou_all)
  #printresults("CC", cc_all)
  printresults("Kappa ", kappa_all)
  printresults("Precision", precision_all)
  printresults("Recall", recall_all)
  return dice_all, iou_all, precision_all, recall_all

def apply_metrics_altamira(path_result, path_gt):
  dice_all = []
  #f1_all = []
  #rmse_all = []
  iou_all = []
  #cc_all = []
  #mre_all = []
  kappa_all = []
  precision_all = []
  recall_all = []


  for i in range(1, 50):
    #print(path_result + str(i) + '.png')
    img = load_bin(path_result + str(i) + '.png')
    gt = load_bin(path_gt + 'Altamira_2020_' + str(i) + '/cm/cm_gt.png')
    dice_all.append(dice(gt, img))
    #f1_all.append(f1_score(gt.flatten(), img.flatten()))
    #rmse_all.append(rmse(gt, img))
    iou_all.append(calculate_iou(gt, img))
    #corr, _ = pearsonr(gt.flatten(), img.flatten())
    #cc_all.append(corr)
    kappa_all.append(cohen_kappa_score(gt.flatten(), img.flatten()))
    precision_all.append(precision(img, gt))
    recall_all.append(recall(img, gt))

  printresults("DICE", dice_all)
  #printresults("F1 Score", f1_all)
  #printresults("RMSE", rmse_all)
  printresults("IoU", iou_all)
  #printresults("CC", cc_all)
  printresults("Kappa ", kappa_all)
  printresults("Precision", precision_all)
  printresults("Recall", recall_all)
  return dice_all, iou_all, precision_all, recall_all

def apply_metrics_altamira_yolo(path_result, path_gt):
  dice_all = []
  iou_all = []
  kappa_all = []
  precision_all = []
  recall_all = []


  for i in range(1, 50):
    #print(path_result + str(i) + '.png')
    img = load_bin(path_result + str(i) + '_mask.png')
    gt = load_bin(path_gt + str(i) + '/cm/cm_gt.png')
    dice_all.append(dice(gt, img))
    #f1_all.append(f1_score(gt.flatten(), img.flatten()))
    #rmse_all.append(rmse(gt, img))
    iou_all.append(calculate_iou(gt, img))
    #corr, _ = pearsonr(gt.flatten(), img.flatten())
    #cc_all.append(corr)
    kappa_all.append(cohen_kappa_score(gt.flatten(), img.flatten()))
    precision_all.append(precision(img, gt))
    recall_all.append(recall(img, gt))

  printresults("DICE", dice_all)
  #printresults("F1 Score", f1_all)
  #printresults("RMSE", rmse_all)
  printresults("IoU", iou_all)
  #printresults("CC", cc_all)
  printresults("Kappa ", kappa_all)
  printresults("Precision", precision_all)
  printresults("Recall", recall_all)
  return dice_all, iou_all, precision_all, recall_all


def load_bin_255(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    return img


def salve_compare(path_result, path_gt, path_save, name):
    # List files in path_gt
    entries = os.listdir(path_gt)
    files = [f for f in entries if os.path.isfile(os.path.join(path_gt, f))]

    for i in files:
        img = load_bin_255(os.path.join(path_result, i))
        gt = load_bin_255(os.path.join(path_gt, i))
        cp_img = np.stack((gt, img, gt), 2)

        # Create a new filename with a prefix
        new_filename = f"{name}_{i}"
        save_path = os.path.join(path_save, new_filename)

        # Save
        cv2.imwrite(save_path, cp_img)

