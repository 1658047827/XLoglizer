from enum import Enum


class LabelType(str, Enum):
    NEXT_LOG = "next_log"
    ANOMALY = "anomaly"


class FeatureType(str, Enum):
    SEQUENTIAL = "sequential"
    SEMANTIC = "semantic"


class WindowType(str, Enum):
    SLIDING = "sliding"
    SESSION = "session"


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

    def compute_num_window(self, seq_len):
        # if seq_len <= self.window_size:
        #     return 0
        # else:
        return 1 + (seq_len - 1 - self.window_size) // self.stride

    def sliding_window(self, session_dict):
        for session_id, data in session_dict.items():
            i = 0
            windows = []
            labels = []
            eids = data["eids"]
            session_len = len(eids)
            while i + self.window_size < session_len:
                windows.append(eids[i : i + self.window_size])
                labels.append(eids[i + self.window_size])
                i += self.stride
            # if i == 0:
            #     eids.extend([0] * (self.window_size - session_len))
            #     inputs.append(eids)
            #     outputs.append(0)  

            session_dict[session_id]["windows"] = windows
            session_dict[session_id]["labels"] = labels
        
        return session_dict

    def fit(self, eid2template):
        self.eid2template = eid2template
        self.eid2template[0] = "padding"

        if self.label_type == LabelType.NEXT_LOG:
            self.meta_data["num_labels"] = len(self.eid2template)
        elif self.label_type == LabelType.ANOMALY:
            self.meta_data["num_labels"] = 2

    def transform(self, session_dict):
        return self.sliding_window(session_dict)
