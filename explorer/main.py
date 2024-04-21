import os
import sys
import pickle
import joblib
import argparse
import json
import torch
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader

file_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(file_dir)
sys.path.append(base_dir)

from loglizer.models import DeepLog
from loglizer.utils import *
from loglizer.feature import *
from loglizer.detect import DetectGranularity
from loglizer.dataset import *

from approach import DeepStellar


parser = argparse.ArgumentParser()
parser.add_argument("--reduced_dim", default=32, type=int)
parser.add_argument("--state_num", default=40, type=int)
parser.add_argument("--record_id", default="20240410000119", type=str)
args = parser.parse_args()


def load_config(file_path):
    with open(file_path, "r") as fr:
        config_dict = json.load(fr)
    config_dict["label_type"] = LabelType(config_dict["label_type"])
    config_dict["feature_type"] = FeatureType(config_dict["feature_type"])
    config_dict["window_type"] = WindowType(config_dict["window_type"])
    config_dict["detect_granularity"] = DetectGranularity(
        config_dict["detect_granularity"]
    )
    return argparse.Namespace(**config_dict)


if __name__ == "__main__":
    record_id = datetime.now().strftime("%Y%m%d%H%M%S")
    setup_logger("explorer", f"{file_dir}/logs/{record_id}.log")
    logging.getLogger("explorer").info(args)
    config = load_config(f"{base_dir}/loglizer/configs/{args.record_id}.json")
    seed_everything(config.seed)

    with open(
        f"{base_dir}/loglizer/data/{config.data_dir}/eid2template.pkl", "rb"
    ) as fr:
        eid2template = pickle.load(fr)

    extractor = FeatureExtractor(
        config.label_type,
        config.feature_type,
        config.window_type,
        config.window_size,
        config.stride,
    )
    extractor.fit(eid2template)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepLog(
        config.input_size,
        config.hidden_size,
        config.num_layers,
        extractor.meta_data["num_labels"],
    ).to(device)
    model.load_state_dict(
        torch.load(f"{base_dir}/loglizer/checkpoints/{args.record_id}.pth")
    )
    model.eval()

    with open(
        f"{base_dir}/loglizer/data/{config.data_dir}/session_train.pkl", "rb"
    ) as fr:
        session_train = pickle.load(fr)
    dataset_train = LogDataset(extractor.transform(session_train))
    dataloader_train = DataLoader(
        dataset_train,
        config.batch_size,
        collate_fn=log_collate,
        pin_memory=True,
    )

    deepstellar = DeepStellar(
        model,
        device,
        config.window_size,
        config.input_size,
        config.hidden_size,
        extractor.meta_data["num_labels"],
        args.reduced_dim,
        args.state_num,
    )
    df = deepstellar.profile(dataloader_train, config.topk)
    vectors = np.load(f"{file_dir}/cache/vectors.npy")
    reduced_vectors = deepstellar.dimension_reduction(vectors)
    traces = deepstellar.state_abstraction(reduced_vectors)

    df["trace"] = traces.tolist()
    df["index"] = df.index
    df.reset_index(drop=True, inplace=True)
    df.to_json(f"{file_dir}/cache/train_dataset.json")

    inputs = np.load(f"{file_dir}/cache/inputs.npy")
    state_input = deepstellar.gather_state_input_statistics(traces, inputs)
    preds = np.load(f"{file_dir}/cache/preds.npy")
    state_label = deepstellar.gather_state_label_statistics(traces, preds)
    fidelity = deepstellar.fidelity(traces, preds, state_label)
    transitions = deepstellar.get_transitions(traces)
    transitions = np.load(f"{file_dir}/cache/transitions.npy")
    state_label = np.load(f"{file_dir}/cache/state_label.npy")
    # ########################################
    # # threshold: principles of statistics? #
    # ########################################
    deepstellar.get_graph(transitions, state_input, 0.01)

    joblib.dump(deepstellar, f"{file_dir}/save/{record_id}.joblib")
