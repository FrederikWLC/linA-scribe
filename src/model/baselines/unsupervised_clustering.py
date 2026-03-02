import cv2
import numpy as np

from model.scribe import Scribe

from model.scribe import Scribe
import cv2
import numpy as np


class UnsupervisedClustering(Scribe):
    def __init__(self, n_clusters=2):
        self.n_clusters = n_clusters

    def scribe(self, image):

        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        h, w = image.shape[:2]

        pixel_values = image.reshape((-1, 3)).astype(np.float32)

        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            100,
            0.2
        )

        _, labels, centers = cv2.kmeans(
            pixel_values,
            self.n_clusters,
            None,
            criteria,
            10,
            cv2.KMEANS_RANDOM_CENTERS
        )

        labels = labels.reshape((h, w))

        # Decide foreground as darker cluster (common for inscriptions)
        centers_gray = np.mean(centers, axis=1)
        foreground_label = np.argmin(centers_gray)
        mask = (labels == foreground_label).astype(np.uint8) * 255
        return cv2.bitwise_not(mask)