import os
import sys
import json
import torch
import joblib
import pickle
import argparse
import pandas as pd
import torch.nn.functional as F
from flask import Flask, request
from flask_cors import CORS

file_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(file_dir)
sys.path.append(base_dir)
sys.path.append(f"{base_dir}/explorer")

from loglizer.models import DeepLog
from loglizer.feature import *

parser = argparse.ArgumentParser()
parser.add_argument("--explorer", default="20240421024529", type=str)
parser.add_argument("--loglizer", default="20240410000119", type=str)
args = parser.parse_args()


class Predictor:
    def __init__(self):
        self.loglizer = None
        self.explorer = None
        with open(f"{base_dir}/loglizer/configs/{args.loglizer}.json", "r") as fr:
            config_dict = json.load(fr)
        self.config = argparse.Namespace(**config_dict)

    def load_loglizer(self):
        with open(
            f"{base_dir}/loglizer/data/{self.config.data_dir}/eid2template.pkl", "rb"
        ) as fr:
            eid2template = pickle.load(fr)

        extractor = FeatureExtractor(
            self.config.label_type,
            self.config.feature_type,
            self.config.window_type,
            self.config.window_size,
            self.config.stride,
        )
        extractor.fit(eid2template)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loglizer = DeepLog(
            self.config.input_size,
            self.config.hidden_size,
            self.config.num_layers,
            extractor.meta_data["num_labels"],
        ).to(self.device)
        self.loglizer.load_state_dict(
            torch.load(f"{base_dir}/loglizer/checkpoints/{args.loglizer}.pth")
        )
        self.loglizer.eval()

    def load_explorer(self):
        self.explorer = joblib.load(f"{base_dir}/explorer/save/{args.explorer}.joblib")

    def predict(self, input):
        x = (
            torch.tensor(input, dtype=torch.float)
            .view(-1, self.config.window_size, self.config.input_size)
            .to(self.device)
        )
        out, pred = self.loglizer.profile(x)
        # print(out.shape, pred.shape)
        last = pred[:, -1, :].squeeze()
        topk_values, topk_indices = torch.topk(last, self.config.topk)
        topk_pred = [
            {"name": f"E{idx}", "value": value.item()}
            for idx, value in zip(topk_indices, topk_values)
        ]
        topk_pred.append(
            {"name": "Others", "value": 1.0 - torch.sum(topk_values).item()}
        )

        sample = out.squeeze().cpu().numpy()
        reduced_vec = self.explorer.reducer.transform(sample)
        trace = self.explorer.abstractor.predict(reduced_vec)
        return topk_pred, trace.tolist()

    def sliding_window(self, eids):
        i = 0
        windows = []
        labels = []
        session_len = len(eids)
        while i + self.config.window_size < session_len:
            windows.append(eids[i : i + self.config.window_size])
            labels.append(eids[i + self.config.window_size])
            i += self.config.stride
        if i == 0:
            eids.extend([0] * (self.config.window_size - session_len))
            windows.append(eids)
            labels.append(0)
        return {"windows": windows, "labels": labels}

    def detect(self, session_dict):
        x = torch.tensor(session_dict["windows"], dtype=torch.float).to(self.device)
        y = torch.tensor(session_dict["labels"]).to(self.device)
        pred = self.loglizer(
            x.view(-1, self.config.window_size, self.config.input_size)
        )
        # Compute loss
        probs = torch.gather(torch.softmax(pred, dim=1), 1, y.unsqueeze(1))
        sample_losses = -torch.log(probs.squeeze())
        losses = sample_losses.tolist()
        # print("Sample losses:", losses)

        topk_values, topk_indices = torch.topk(pred, self.config.topk)

        return losses, topk_values.tolist(), topk_indices.tolist()


if __name__ == "__main__":
    predictor = Predictor()
    predictor.load_loglizer()
    predictor.load_explorer()
    df = pd.read_json(f"{file_dir}/data/train_dataset.json")

    app = Flask("XLoglizer", static_folder=f"{file_dir}/static")
    CORS(app, resources={r"/*": {"origins": "http://localhost:5173"}})

    @app.route("/detect", methods=["POST"])
    def detect():
        session = request.json.get("session")
        session_dict = predictor.sliding_window(session)
        losses, topk_values, topk_indices = predictor.detect(session_dict)
        return {
            "losses": losses,
            "topk_preds": topk_indices,
            "topk_values": topk_values,
        }

    @app.route("/predict", methods=["POST"])
    def predict():
        input = request.json.get("data")
        last_topk_pred, trace = predictor.predict(input)
        return {"topk_pred": last_topk_pred, "trace": trace}

    @app.route("/dataset", methods=["GET"])
    def dataset():
        page = int(request.args.get("page"))
        size = int(request.args.get("size"))
        total = df.shape[0]
        end = page * size if page * size <= total else total
        selected = df.iloc[(page - 1) * size : end]
        data = selected.to_dict(orient="records")
        return {"data": data, "total": total}

    @app.route("/static/<path:filename>")
    def static_file(filename):
        return app.send_static_file(filename)

    app.run()
