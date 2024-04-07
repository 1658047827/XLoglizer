import torch
import pandas as pd
from collections import defaultdict
from torch.utils.data import Dataset


class LogDataset(Dataset):
    def __init__(self, session_dict):
        # data = defaultdict(list)
        data = []
        for session_id, data_dict in session_dict.items():
            anomaly = data_dict["anomaly"]
            for window, label in zip(data_dict["windows"], data_dict["labels"]):
                # data["session_id"].append(session_id)
                # data["feature"].append(window)
                # data["label"].append(label)
                # data["anomaly"].append(anomaly)
                data.append({
                    "session_id": session_id,
                    "feature": window,
                    "label": label,
                    "anomaly": anomaly,
                })
        self.data = data

    def __len__(self):
        # return len(self.data["session_id"])
        return len(self.data)

    def __getitem__(self, index):
        # return {
        #     "session_id": self.data["session_id"][index],
        #     "feature": self.data["feature"][index],
        #     "label": self.data["label"][index],
        #     "anomaly": self.data["anomaly"][index],
        # }
        return self.data[index]


def log_collate(batch):
    return {
        "session_id": [sample["session_id"] for sample in batch],
        "feature": torch.tensor([sample["feature"] for sample in batch], dtype=torch.float),
        "label": torch.tensor([sample["label"] for sample in batch]),
        "anomaly": torch.tensor([sample["anomaly"] for sample in batch]),
    }
