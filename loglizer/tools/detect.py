import torch
import logging
import tracemalloc
import pandas as pd
from tqdm import tqdm
from enum import Enum
from collections import defaultdict
from sklearn.metrics import confusion_matrix


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

    @torch.no_grad()
    def predict(self, test_loader):
        tracemalloc.start()  # 开始跟踪内存分配

        self.model.eval()
        store = defaultdict(list)
        preds_topk = []

        for batch in tqdm(test_loader):
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

        actual = (session_df["anomaly"] > 0).astype(int)
        for g in range(1, self.topk + 1):
            pred = (session_df[f"pred_{g}"] > 0).astype(int)

            cm = confusion_matrix(actual, pred)
            tn, fp, fn, tp = cm.ravel()
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = 2 * precision * recall / (precision + recall)
            accuracy = (tp + tn) / (tp + fp + fn + tn)
            
            logging.getLogger("loglizer").info({
                "topk": g,
                "false positive": fp,
                "false negative": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "accuracy": accuracy,
            })

        snapshot = tracemalloc.take_snapshot()  # 获取内存快照
        top_stats = snapshot.statistics('lineno')
        for stat in top_stats:
            print(stat)