import os
import re
import json
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict

if __name__ == "__main__":
    file_dir = os.path.dirname(os.path.abspath(__file__))
    log_file = f"{file_dir}/HDFS_2k.log_structured.csv"
    label_file = f"{file_dir}/anomaly_label.csv"
    template_file = f"{file_dir}/HDFS_2k.log_templates.csv"
    dump_dir = f"{file_dir}/../../loglizer/data/HDFS_2k"
    shuffle = True
    valid_ratio = 0.1
    test_ratio = 0.1
    os.makedirs(dump_dir, exist_ok=True)

    logs_df = pd.read_csv(log_file, engine="c", na_filter=False, memory_map=True)
    label_df = pd.read_csv(label_file, engine="c", na_filter=False, memory_map=True)
    templates_df = pd.read_csv(template_file, engine="c", na_filter=False, memory_map=True)

    eid2template = {}
    for _, row in templates_df.iterrows():
        event_id = int(row["EventId"][1:])
        eid2template[event_id] = row["EventTemplate"]
    with open(f"{dump_dir}/eid2template.pkl", "wb") as fw:
        pickle.dump(eid2template, fw)

    label_df["Label"] = label_df["Label"].map({"Anomaly": 1, "Normal": 0})
    blk_label = label_df.set_index("BlockId")["Label"].to_dict()

    session_dict = defaultdict(lambda: {"eids": [], "anomaly": 0})
    for _, row in logs_df.iterrows():
        # Extract all blk_id from this log and use a set to deduplicate.
        blkId_set = set(re.findall(r"(blk_-?\d+)", row["Content"]))
        for blkId in blkId_set:
            session_dict[blkId]["eids"].append(int(row["EventId"][1:]))
    for blkId in session_dict.keys():
        session_dict[blkId]["anomaly"] = blk_label[blkId]

    session_ids = [key for key in session_dict.keys()]
    if shuffle:
        np.random.shuffle(session_ids)

    normal_ids = [k for k in session_ids if session_dict[k]["anomaly"] == 0]
    anomaly_ids = [k for k in session_ids if session_dict[k]["anomaly"] == 1]

    train_num = int((1 - valid_ratio - test_ratio) * len(session_ids))
    valid_num = int(valid_ratio * len(session_ids))
    test_num = len(session_ids) - train_num - valid_num

    session_ids_train = normal_ids[:train_num]
    session_ids_valid = normal_ids[train_num : train_num + valid_num]
    session_ids_test = normal_ids[train_num + valid_num :] + anomaly_ids
    np.random.shuffle(session_ids_test)

    session_train = {k: session_dict[k] for k in session_ids_train}
    session_valid = {k: session_dict[k] for k in session_ids_valid}
    session_test = {k: session_dict[k] for k in session_ids_test}

    with open(f"{dump_dir}/session_train.pkl", "wb") as fw:
        pickle.dump(session_train, fw)
    with open(f"{dump_dir}/session_valid.pkl", "wb") as fw:
        pickle.dump(session_valid, fw)
    with open(f"{dump_dir}/session_test.pkl", "wb") as fw:
        pickle.dump(session_test, fw)

    # Generate json description for processed data.
    description = {
        "total_num": len(session_ids),
        "normal_num": len(normal_ids),
        "anomaly_num": len(anomaly_ids),
        "shuffle": shuffle,
        "train": {
            "ratio": 1 - valid_ratio - test_ratio,
            "num": train_num,
            "normal": train_num,
            "anomaly": 0,
        },
        "valid": {
            "ratio": valid_ratio,
            "num": valid_num,
            "normal": valid_num,
            "anomaly": 0,
        },
        "test": {
            "ratio": test_ratio,
            "num": test_num,
            "normal": test_num - len(anomaly_ids),
            "anomaly": len(anomaly_ids),
        },
    }
    with open(f"{dump_dir}/description.json", "w") as fw:
        json.dump(description, fw, indent=4)