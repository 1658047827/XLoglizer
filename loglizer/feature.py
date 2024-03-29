from enum import Enum


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
