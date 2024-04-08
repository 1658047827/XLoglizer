import torch
import logging
import numpy as np
import pandas as pd
from enum import Enum
from collections import defaultdict
from sklearn.metrics import precision_score, f1_score, recall_score, accuracy_score


class DetectGranularity(str, Enum):
    SESSION = "session"
    WINDOW = "window"


class Detector:
    def __init__(
        self,
        model,
        device,
        window_size,
        input_size,
        topk,
        detect_granularity,
    ):
        self.model = model
        self.device = device
        self.window_size = window_size
        self.input_size = input_size
        self.topk = topk
        self.detect_granularity = detect_granularity

    def predict(self, test_loader):
        self.model.eval()
        store = defaultdict(list)
        preds_topk = []

        with torch.no_grad():
            for batch in test_loader:
                x = batch["feature"].view(-1, self.window_size, self.input_size).to(self.device)
                y = batch["label"].to(self.device)
                output = self.model(x)

                _, topk_indices = torch.topk(output, self.topk)
                topk_matches = topk_indices.eq(y.view(-1, 1)).int()
                acc_matches = torch.cumsum(topk_matches, dim=1) > 0
                preds = (~acc_matches).int().cpu().numpy()

                store["session_id"].extend(batch["session_id"])
                store["anomaly"].extend(batch["anomaly"])
                preds_topk.extend(preds)

        df = pd.DataFrame(store)
        topk_df = pd.DataFrame(preds_topk)
        for g in range(1, self.topk + 1):
            df[f"pred_{g}"] = topk_df[g - 1]

        if self.detect_granularity == DetectGranularity.SESSION:
            session_df = df.groupby("session_id", as_index=False).sum()
            print(session_df)

        actual = (session_df["anomaly"] > 0).astype(int)
        for g in range(1, self.topk + 1):
            pred = (session_df[f"pred_{g}"] > 0).astype(int)

            precision = precision_score(actual, pred)
            recall = recall_score(actual, pred)
            f1 = f1_score(actual, pred)
            accuracy = accuracy_score(actual, pred)

            logging.getLogger("loglizer").info({
                "topk": g,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "accuracy": accuracy 
            })
