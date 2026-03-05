import torch
from torchmetrics.classification import (
    BinaryAccuracy, BinaryPrecision, BinaryRecall, BinarySpecificity, BinaryF1Score
)
import cv2
import numpy as np

class Evaluator:

    def evaluate(self, X: list, Y: list, tolerance=0):
        f1 = BinaryF1Score()
        acc  = BinaryAccuracy()
        prec = BinaryPrecision()
        rec  = BinaryRecall()
        spec = BinarySpecificity()
        metrics = dict(Accuracy=acc, Precision=prec, Recall=rec, Specificity=spec, F1=f1)
        results = self.compute_metrics(X, Y, metrics, tolerance)
        results["IoU"] = results["F1"] / (2-results["F1"]) # IoU can be derived from F1
        
        return results
    
    def compute_metrics(self,X,Y,metrics,tolerance=0):
        intermediate_results = {name: [] for name in metrics.keys()}
        for name, _ in metrics.items():
            intermediate_results.setdefault(name, [])

        for y_hat_t, y_t in zip(*self.get_tensored_preds_and_labels(X, Y, tolerance)):

            for name, metric in metrics.items():
                metric.reset()
                intermediate_results[name].append(metric(y_hat_t, y_t))

        results = {name: torch.stack(intermediate_results[name]).mean().item() for name in metrics.keys()}
        return results
    
    def get_tensored_preds_and_labels(self, X, Y, tolerance=0):
        Y_hat_t = []
        Y_t = []
        for x, y in zip(X, Y):
            
            if tolerance > 0:
                y = dilate_mask(y, tolerance)
                y_hat = dilate_mask(self.scribe(x), tolerance)
            
            y = torch.from_numpy(y).squeeze()
            y = (y > 127).long()
            y = 1 - y

            y_hat = torch.from_numpy(y_hat).squeeze()
            y_hat = (y_hat > 127).long()
            y_hat = 1 - y_hat

            Y_hat_t.append(y_hat.unsqueeze(0))
            Y_t.append(y.unsqueeze(0))

        return Y_hat_t, Y_t

    def scribe(self, x): 
        pass #stand in function to be defined in Scribe class



def dilate_mask(mask: np.ndarray, r: int) -> np.ndarray:
    if r <= 0:
        return mask
    
    mask = cv2.bitwise_not(mask)  # Invert mask to dilate ink
    kernel = np.ones((2*r + 1, 2*r + 1), np.uint8)
    dilated = cv2.dilate(mask.astype(np.uint8), kernel)
    dilated = cv2.bitwise_not(dilated)  # Invert back to original format

    return dilated