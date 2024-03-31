import torch
from enum import Enum
from torch.utils.data import TensorDataset


class LabelType(Enum):
    NEXT_LOG = 1
    ANOMALY = 2


class FeatureType(Enum):
    SEQUENTIAL = 1
    SEMANTIC = 2


class WindowType(Enum):
    SLIDING = 1


class FeatureExtractor:
    def __init__(
        self,
        label_type=LabelType.NEXT_LOG,
        feature_type=FeatureType.SEQUENTIAL,
        window_type=WindowType.SLIDING,
        window_size=10,
        stride=1,
    ):
        self.label_type = label_type
        self.feature_type = feature_type
        self.window_type = window_type
        self.window_size = window_size
        self.stride = stride
        self.meta_data = {}

    def generate(self, session_dict):
        inputs = []
        outputs = []
        for session_id, data in session_dict.items():
            i = 0
            eids = data["eids"]
            session_len = len(eids)
            while i + self.window_size < session_len:
                inputs.append(eids[i : i + self.window_size])
                outputs.append(eids[i + self.window_size])
                i += self.stride
            if i == 0:
                pass
        return TensorDataset(torch.tensor(inputs), torch.tensor(outputs))

    def fit(self, eid2template):
        self.eid2template = eid2template
        self.eid2template[0] = "padding"

        if self.label_type == LabelType.NEXT_LOG:
            self.meta_data["num_labels"] = len(self.eid2template)
        elif self.label_type == LabelType.ANOMALY:
            self.meta_data["num_labels"] = 2

    def transform(self, session_dict, save=False):
        return self.generate(session_dict)
