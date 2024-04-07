import os
import pickle
import torch
from datetime import datetime
from torch.utils.data import DataLoader
from feature import FeatureExtractor
from tools.train import Trainer
from tools.detect import Detector
from tools.utils import setup_logger, seed_everything
from models import DeepLog
from args import Args

file_dir = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    record_id = datetime.now().strftime("%Y%m%d%H%M%S")
    setup_logger(f"{file_dir}/logs/{record_id}.log")
    a = Args()
    a.dump_args(f"{record_id}.json")
    args = a.get_args()
    seed_everything(args["seed"])

    with open(f"{file_dir}/data/{args['data_dir']}/eid2template.pkl", "rb") as fr:
        eid2template = pickle.load(fr)

    extractor = FeatureExtractor(
        args["label_type"],
        args["feature_type"],
        args["window_type"],
        args["window_size"],
        args["stride"],
    )
    extractor.fit(eid2template)

    with open(f"{file_dir}/data/{args['data_dir']}/session_train.pkl", "rb") as fr:
        session_train = pickle.load(fr)
    with open(f"{file_dir}/data/{args['data_dir']}/session_valid.pkl", "rb") as fr:
        session_valid = pickle.load(fr)

    dataset_train = extractor.transform(session_train)
    dataset_valid = extractor.transform(session_valid)

    dataloader_train = DataLoader(dataset_train, args["batch_size"], shuffle=True, pin_memory=True)
    dataloader_valid = DataLoader(dataset_valid, 4096, shuffle=False, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    deeplog = DeepLog(
        args["input_size"],
        args["hidden_size"],
        args["num_layers"],
        extractor.meta_data["num_labels"],
    ).to(device)
    optimizer = torch.optim.Adam(deeplog.parameters(), args["learning_rate"])
    criterion = torch.nn.CrossEntropyLoss()

    trainer = Trainer(
        deeplog,
        device,
        optimizer,
        criterion,
        args["window_size"],
        args["input_size"],
    )
    trainer.fit(dataloader_train, dataloader_valid, args["epochs"])

    with open(f"{file_dir}/data/{args['data_dir']}/session_test.pkl", "rb") as fr:
        session_test = pickle.load(fr)

    dataset_test = extractor.transform(session_test)

    dataloader_test = DataLoader(dataset_test, 4096, shuffle=True, pin_memory=True)

    detector = Detector(
        deeplog, 
        device, 
        args["window_size"], 
        args["input_size"], 
        args["topk"],
    )
    detector.predict(dataloader_test)
