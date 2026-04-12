import math
import statistics as stats
from utils.binary_mask import BinaryMask
from model.scribe import predict_batch
import numpy as np


# F1 / Dice score metric
def BinaryDiceScore(y_hat: BinaryMask, y: BinaryMask) -> float:
    intersection = BinaryMask.from_intersection(y_hat, y).sum()
    dice = 2*intersection / (y_hat.sum() + y.sum()) if (y_hat.sum() + y.sum()) > 0 else 0
    return float(dice)

# Intersection over Union metric
def BinaryIoU(y_hat: BinaryMask, y: BinaryMask) -> float:
    intersection = BinaryMask.from_intersection(y_hat, y).sum()
    union = BinaryMask.from_union(y_hat, y).sum()
    return float(intersection / union if union > 0 else 0)

# True Positives: the number of pixels correctly predicted as foreground (ink)
def tp(y_hat: BinaryMask, y: BinaryMask) -> int:
    # equals the intersection of predicted and actual foreground (ink) pixels
    return BinaryMask.from_intersection(y_hat, y).sum()

# False Positives: the number of pixels incorrectly predicted as foreground (ink)
def fp(y_hat: BinaryMask, y: BinaryMask) -> int:
    # equals the intersection of predicted foreground (ink) and actual background
    return BinaryMask.from_intersection(y_hat, y.invert()).sum()

# False negatives: the number of pixels incorrectly predicted as background 
def fn(y_hat: BinaryMask, y: BinaryMask) -> int:
    # equals the number of pixels predicted as background that are actually foreground (ink)
    return BinaryMask.from_intersection(y_hat.invert(), y).sum()

# True Negatives: the number of pixels correctly predicted as background
def tn(y_hat: BinaryMask, y: BinaryMask) -> int:
    # equals the intersection of the inverted masks, which includes size of correctly predicted background
    return BinaryMask.from_intersection(y_hat.invert(), y.invert()).sum()

# Accuracy metric
def BinaryAccuracy(y_hat: BinaryMask, y: BinaryMask) -> float:
    # Accuracy = TP + TN / (TP + FP + FN + TN)
    accuracy = (tp(y_hat, y) + tn(y_hat, y)) / (tp(y_hat, y) + fp(y_hat, y) + fn(y_hat, y) + tn(y_hat, y)) if (tp(y_hat, y) + fp(y_hat, y) + fn(y_hat, y) + tn(y_hat, y)) > 0 else 0
    return float(accuracy)

# Precision metric
def BinaryPrecision(y_hat: BinaryMask, y: BinaryMask) -> float:
    # Precision = TP / (TP + FP)
    precision = tp(y_hat, y) / (tp(y_hat, y) + fp(y_hat, y)) if (tp(y_hat, y) + fp(y_hat, y)) > 0 else 0
    return float(precision)

# Recall metric
def BinaryRecall(y_hat: BinaryMask, y: BinaryMask) -> float:
    # Recall = TP / (TP + FN)
    recall = tp(y_hat, y) / (tp(y_hat, y) + fn(y_hat, y)) if (tp(y_hat, y) + fn(y_hat, y)) > 0 else 0
    return float(recall)

# Specificity metric
def BinarySpecificity(y_hat: BinaryMask, y: BinaryMask) -> float:
    # Specificity = TN / (TN + FP)
    specificity = tn(y_hat, y) / (tn(y_hat, y) + fp(y_hat, y)) if (tn(y_hat, y) + fp(y_hat, y)) > 0 else 0
    return float(specificity)


METRICS = dict(dice=BinaryDiceScore)

# Compute metrics for each datapoint
# given predictions and ground truths
def compute_metrics(Y_hat: list[BinaryMask], Y: list[BinaryMask], metrics: dict[str, callable] = METRICS) -> dict[str, list[float]]:
    results = {}
    for name, metric in metrics.items():
        values = []
        for y_hat, y in zip(Y_hat,Y):
            values.append(metric(y_hat, y))
        results[name] = values
    return results

# Compute statistical summaries for computed metrics
# given metric results
def summarize_results(results: dict[str, list[float]]) -> dict[str, float]:
    resume = {}
    for name, values in results.items():
        median = stats.median(values) # sample median
        mean = stats.mean(values) # sample mean
        std = stats.stdev(values) # sample standard deviation
        n = len(values) # number of sample observations
        std_error = std / math.sqrt(n) # sample standard error
        resume.update({name+"_"+key : value for key, value in {'median': median, 'mean': mean, 'std': std, 'std_error': std_error,'n':n}.items()})
    return resume

# Evaluate a model on a dataset
# and return the results and a statistical resume of metrics
def evaluate_model(model, X: list[np.ndarray], Y: list[BinaryMask], metrics=METRICS) -> tuple[dict[str, list[float]], dict[str, float]]:
    Y_hat = predict_batch(model, X)
    Y = [BinaryMask.from_image(gt) for gt in Y]
    results = compute_metrics(Y_hat, Y, metrics)
    resume = summarize_results(results)
    return results,resume