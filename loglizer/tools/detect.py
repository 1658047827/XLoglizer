import torch
import numpy as np
from sklearn.metrics import confusion_matrix


class Detector:
    def __init__(self, model, device, window_size, input_size, topk):
        self.model = model
        self.device = device
        self.window_size = window_size
        self.input_size = input_size
        self.topk = topk

    def predict(self, test_loader):
        self.model.eval()
        cm = np.zeros((2, 2), dtype=int)
        with torch.no_grad():
            for x, y, anomaly in test_loader:
                x = x.view(-1, self.window_size, self.input_size).to(self.device)
                y = y.to(self.device)
                output = self.model(x)

                _, topk_indices = torch.topk(output, self.topk)
                topk_matches = topk_indices.eq(y.view(-1, 1)).any(dim=1)
                pred = (~topk_matches).int().cpu()
                cm += confusion_matrix(anomaly, pred)

        print(cm)
        tp = cm[1, 1]
        fp = cm[0, 1]
        fn = cm[1, 0]

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)

        print(f"precision: {precision}")
        print(f"recall: {recall}")
        print(f"f1 score: {f1}")
