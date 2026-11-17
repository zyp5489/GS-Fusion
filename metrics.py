import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, precision_score,
    recall_score, f1_score, auc, accuracy_score, matthews_corrcoef
)
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, median_absolute_error,
    mean_absolute_percentage_error, max_error, r2_score
)
from scipy.stats import pearsonr


def concordance_index(y_true, y_pred):
    """计算一致性指数 (C-index)"""
    summ = 0
    pair = 0
    for i in range(1, len(y_true)):
        for j in range(0, i):
            if i != j:
                if y_true[i] > y_true[j]:
                    pair += 1
                    summ += 1 * (y_pred[i] > y_pred[j]) + 0.5 * (y_pred[i] == y_pred[j])
    return summ / pair if pair != 0 else 0.0


def r_squared_extra(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = np.full_like(y_obs, np.mean(y_obs))
    y_pred_mean = np.full_like(y_pred, np.mean(y_pred))
    mult = np.sum((y_pred - y_pred_mean) * (y_obs - y_obs_mean)) ** 2
    y_obs_sq = np.sum((y_obs - y_obs_mean) ** 2)
    y_pred_sq = np.sum((y_pred - y_pred_mean) ** 2)
    return mult / float(y_obs_sq * y_pred_sq)


def slope(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    return np.sum(y_obs * y_pred) / float(np.sum(y_pred * y_pred))


def r_squared_zero(y_obs, y_pred):
    k = slope(y_obs, y_pred)
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = np.full_like(y_obs, np.mean(y_obs))
    upp = np.sum((y_obs - k * y_pred) ** 2)
    down = np.sum((y_obs - y_obs_mean) ** 2)
    return 1 - (upp / float(down))


def rm2_score(y_orig, y_pred):
    r2 = r_squared_extra(y_orig, y_pred)
    r02 = r_squared_zero(y_orig, y_pred)
    return r2 * (1 - np.sqrt(np.abs((r2 * r2) - (r02 * r02))))


def regression_metrics(y_true, y_pred, prefix="", include_rm2=False, include_ci=False):
    """回归任务常用指标"""
    metrics = {}
    metrics[f"{prefix}_mse"] = float(mean_squared_error(y_true, y_pred))
    metrics[f"{prefix}_mae"] = float(mean_absolute_error(y_true, y_pred))
    metrics[f"{prefix}_medae"] = float(median_absolute_error(y_true, y_pred))
    metrics[f"{prefix}_mape"] = float(mean_absolute_percentage_error(y_true, y_pred))
    metrics[f"{prefix}_maxe"] = float(max_error(y_true, y_pred))
    metrics[f"{prefix}_r2"] = float(r2_score(y_true, y_pred))
    metrics[f"{prefix}_pearsonr"] = pearsonr(y_true.flatten(), y_pred.flatten())[0]
    if include_rm2:
        metrics[f"{prefix}_rm2"] = rm2_score(y_true.flatten().tolist(), y_pred.flatten().tolist())
    if include_ci:
        metrics[f"{prefix}_ci"] = concordance_index(y_true.flatten().tolist(), y_pred.flatten().tolist())
    return metrics


def classification_metrics(y_true, y_pred, prefix="", transform=torch.sigmoid, threshold=0.5):
    """分类任务指标（用于可选的二元分类）"""
    if transform is not None:
        y_pred = transform(y_pred)
    y_pred_lbl = (y_pred >= threshold).type(torch.float32)
    metrics = {}
    metrics["f1"] = float(f1_score(y_true, y_pred_lbl))
    metrics["precision"] = float(precision_score(y_true, y_pred_lbl, zero_division=0))
    metrics["recall"] = float(recall_score(y_true, y_pred_lbl))
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred_lbl))
    metrics["mcc"] = float(matthews_corrcoef(y_true, y_pred_lbl))
    try:
        metrics["rocauc"] = float(roc_auc_score(y_true, y_pred))
    except ValueError:
        metrics["rocauc"] = np.nan
    try:
        precision_list, recall_list, _ = precision_recall_curve(y_true, y_pred)
        metrics["prauc"] = float(auc(recall_list, precision_list))
    except ValueError:
        metrics["prauc"] = np.nan
    return {f"{prefix}_{k}" if prefix else k: v for k, v in metrics.items()}