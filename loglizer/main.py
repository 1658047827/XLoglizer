import os
import pickle
import torch
from feature import FeatureExtractor
from tools.train import Trainer
from models import DeepLog
from torch.utils.data import DataLoader
from tools.utils import set_log

if __name__ == "__main__":
    set_log()
    file_dir = os.path.dirname(os.path.abspath(__file__))

    with open(f"{file_dir}/data/HDFS_v1/eid2template.pkl", "rb") as fr:
        eid2template = pickle.load(fr)

    extractor = FeatureExtractor()
    extractor.fit(eid2template)

    with open(f"{file_dir}/data/HDFS_v1/session_train.pkl", "rb") as fr:
        session_train = pickle.load(fr)
    with open(f"{file_dir}/data/HDFS_v1/session_valid.pkl", "rb") as fr:
        session_valid = pickle.load(fr)

    dataset_train = extractor.transform(session_train)
    dataset_valid = extractor.transform(session_valid)

    dataloader_train = DataLoader(dataset_train, 1024, shuffle=True, pin_memory=True)
    dataloader_valid = DataLoader(dataset_valid, 4096, shuffle=False, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    deeplog = DeepLog(1, 64, 1, extractor.meta_data["num_labels"]).to(device)
    optimizer = torch.optim.Adam(deeplog.parameters(), 0.01)
    criterion = torch.nn.CrossEntropyLoss()

    trainer = Trainer(deeplog, device, optimizer, criterion)
    trainer.fit(dataloader_train, dataloader_valid, 3)
