import os
import argparse
import json
from feature import LabelType, FeatureType, WindowType
from detect import DetectGranularity

file_dir = os.path.dirname(os.path.abspath(__file__))


class Args:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.args = None

        # model args
        self.parser.add_argument("--input_size", default=1, type=int)
        self.parser.add_argument("--hidden_size", default=64, type=int)
        self.parser.add_argument("--num_layers", default=2, type=int)

        # dataset args
        self.parser.add_argument("--dataset", default="HDFS", type=str)
        self.parser.add_argument("--data_dir", default="HDFS_DeepLog", type=str)

        # feature args
        self.parser.add_argument("--label_type", type=LabelType, choices=list(LabelType), default=LabelType.NEXT_LOG)
        self.parser.add_argument("--feature_type", type=FeatureType, choices=list(FeatureType), default=FeatureType.SEQUENTIAL)
        self.parser.add_argument("--window_type", type=WindowType, choices=list(WindowType), default=WindowType.SLIDING)
        self.parser.add_argument("--window_size", default=10, type=int)
        self.parser.add_argument("--stride", default=1, type=int)

        # train args
        self.parser.add_argument("--epochs", default=50, type=int)
        self.parser.add_argument("--batch_size", default=128, type=int)
        self.parser.add_argument("--learning_rate", default=0.01, type=float)

        # detect args
        self.parser.add_argument("--topk", default=9, type=int)
        self.parser.add_argument("--detect_granularity", type=DetectGranularity, choices=list(DetectGranularity), default=DetectGranularity.SESSION)

        # other args
        self.parser.add_argument("--seed", type=int, default=42)

    def get_args(self):
        if self.args is None:
            self.args = vars(self.parser.parse_args())
        return self.args

    def dump_args(self, file_name):
        if self.args is None:
            self.args = vars(self.parser.parse_args())
        
        with open(f"{file_dir}/configs/{file_name}", "w") as fw:
            json.dump(self.args, fw, indent=4)

    def load_args(self, file_name):
        with open(f"{file_dir}/configs/{file_name}", "r") as fr:
            self.args = json.load(fr)

        self.args["label_type"] = LabelType(self.args["label_type"])
        self.args["feature_type"] = FeatureType(self.args["feature_type"])
        self.args["window_type"] = WindowType(self.args["window_type"])
        self.args["detect_granularity"] = DetectGranularity(self.args["detect_granularity"])