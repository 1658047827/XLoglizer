import os
import sys
import json
import torch
import argparse
from flask import Flask, request

file_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(file_dir)
sys.path.append(base_dir)

from loglizer.models import DeepLog

parser = argparse.ArgumentParser()
parser.add_argument("--explorer", default="20240410000119", type=str)
parser.add_argument("--loglizer", default="20240410000119", type=str)
args = parser.parse_args()


app = Flask("XLoglizer")


class Predictor:
    def __init__(self):
        self.loglizer = None
        self.explorer = None
        with open(f"{base_dir}/loglizer/configs/{args.loglizer}.json", "r") as fr:
            config_dict = json.load(fr)
        self.configs = argparse.Namespace(**config_dict)

    def load_loglizer(self):
        self.loglizer = DeepLog(
            self.configs.input_size,
            self.configs.hidden_size,
            self.configs.num_layers,
            30,  # for HDFS_v1
        )
        self.loglizer.load_state_dict(torch.load(f"{base_dir}/loglizer/checkpoints/{args.loglizer}.pth"))
        self.loglizer.eval()

    def load_explorer(self):
        pass


predictor = Predictor()


@app.before_first_request
def load_models():
    predictor.load_loglizer()
    predictor.load_explorer()


@app.route("/predict", methods=["GET"])
def predict():
    pass


@app.route("/")
def hello():
    return "Hello Flask!"


if __name__ == "__main__":
    app.run()
