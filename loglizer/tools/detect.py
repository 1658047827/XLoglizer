import torch
import numpy as np
import logging
from enum import Enum
from sklearn.metrics import confusion_matrix, precision_score, f1_score, recall_score


class DetectGranularity(str, Enum):
    SESSION = "session"
    WINDOW = "window"


class Detector:
    def __init__(self, model, device, window_size, input_size, topk):
        self.model = model
        self.device = device
        self.window_size = window_size
        self.input_size = input_size
        self.topk = topk

    def predict(self, test_loader):
        self.model.eval()
        cms = np.zeros((self.topk, 2, 2), dtype=int)

        with torch.no_grad():
            for x, y, anomaly in test_loader:
                x = x.view(-1, self.window_size, self.input_size).to(self.device)
                y = y.to(self.device)
                output = self.model(x)

                _, topk_indices = torch.topk(output, self.topk)
                topk_matches = topk_indices.eq(y.view(-1, 1)).int()
                acc_matches = torch.cumsum(topk_matches, dim=1) > 0
                preds = (~acc_matches).int().cpu()

                for i in range(self.topk):
                    cms[i] += confusion_matrix(anomaly, preds[:, i])

        logger = logging.getLogger("loglizer")

        print(cms)
        # tp = cm[1, 1]
        # fp = cm[0, 1]
        # fn = cm[1, 0]

        # precision = tp / (tp + fp)
        # recall = tp / (tp + fn)
        # f1 = 2 * precision * recall / (precision + recall)

        # logger.info(f"precision: {precision}")
        # logger.info(f"recall: {recall}")
        # logger.info(f"f1 score: {f1}")

        # print(precision_score(ano, prd))
        # print(recall_score(ano, prd))
        # print(f1_score(ano, prd))
