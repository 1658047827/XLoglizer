import torch
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset


class LogDataset(Dataset):
    def __init__(self, session_dict):
        data = defaultdict(list)
        for session_id, data_dict in session_dict.items():
            window_num = len(data_dict["labels"])
            anomaly = data_dict["anomaly"]
            data["session_id"].extend([session_id] * window_num)
            data["feature"].extend(data_dict["windows"])
            data["label"].extend(data_dict["labels"])
            data["anomaly"].extend([anomaly] * window_num)

        self.data = data

    def __len__(self):
        return len(self.data["session_id"])

    def __getitem__(self, index):
        return (
            self.data["session_id"][index],
            self.data["feature"][index],
            self.data["label"][index],
            self.data["anomaly"][index],
        )


def log_collate(batch):
    return {
        "session_id": [sample[0] for sample in batch],
        "feature": torch.tensor([sample[1] for sample in batch], dtype=torch.float),
        "label": torch.tensor([sample[2] for sample in batch]),
        "anomaly": [sample[3] for sample in batch],
    }
