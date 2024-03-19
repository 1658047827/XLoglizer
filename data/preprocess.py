import os
import re
import json
import hashlib
import pickle
import pandas as pd
import numpy as np
from collections import defaultdict


def preprocess_hdfs(
    log_file,
    label_file,
    test_ratio=0.2,
    include_anomaly=False,
    shuffle=False,
):
    structured_log = pd.read_csv(log_file, engine="c", na_filter=False, memory_map=True)

    label_data = pd.read_csv(label_file, engine="c", na_filter=False, memory_map=True)
    label_data["Label"] = label_data["Label"].map({"Anomaly": 1, "Normal": 0})
    blk_label = dict(zip(label_data["BlockId"], label_data["Label"]))

    session_dict = defaultdict(lambda: {"templates": [], "anomaly": 0})

    for _, row in structured_log.iterrows():
        # Extract all blk_id from this log and use a set to deduplicate.
        blk_id_set = set(re.findall(r"(blk_-?\d+)", row["Content"]))
        for blk_id in blk_id_set:
            session_dict[blk_id]["templates"].append(row["EventTemplate"])

    for blk_id in session_dict.keys():
        session_dict[blk_id]["anomaly"] = blk_label[blk_id]

    session_id = [key for key in session_dict.keys()]
    if shuffle:
        np.random.shuffle(session_id)

    train_num = int((1 - test_ratio) * len(session_id))
    test_num = int(test_ratio * len(session_id))

    if not include_anomaly:
        normal_id = [key for key in session_id if session_dict[key]["anomaly"] == 0]
        anomaly_id = [key for key in session_id if session_dict[key]["anomaly"] == 1]

        session_id_train = normal_id[0:train_num]
        session_id_test = normal_id[train_num:] + anomaly_id
        np.random.shuffle(session_id_test)
    else:
        session_id_train = session_id[0:train_num]
        session_id_test = session_id[-test_num:]

    # Generate train dataset and test dataset.
    session_train = {k: session_dict[k] for k in session_id_train}
    session_test = {k: session_dict[k] for k in session_id_test}

    session_labels_train = [v["anomaly"] for v in session_train.values()]
    session_labels_test = [v["anomaly"] for v in session_test.values()]

    train_anomaly_rate = sum(session_labels_train) / len(session_labels_train)
    test_anomaly_rate = sum(session_labels_test) / len(session_labels_test)

    # Generate a hash_id for processed data.
    param_list = [log_file, label_file, test_ratio, include_anomaly, shuffle]
    hash_id = hashlib.md5(str(param_list).encode("utf-8")).hexdigest()[0:8]
    os.makedirs(f"./processed/{hash_id}", exist_ok=True)

    with open(f"./processed/{hash_id}/session_train.pkl", "wb") as fw:
        pickle.dump(session_train, fw)
    with open(f"./processed/{hash_id}/session_test.pkl", "wb") as fw:
        pickle.dump(session_test, fw)

    # Generate a json description for processed data.
    description = {
        "log_file": log_file,
        "label_file": label_file,
        "test_ratio": test_ratio,
        "include_anomaly": include_anomaly,
        "shuffle": shuffle,
        "session_num": len(session_id),
        "train_num": train_num,
        "train_anomaly_ratio": train_anomaly_rate,
        "test_num": test_num,
        "test_anomaly_ratio": test_anomaly_rate,
    }
    with open(f"./processed/{hash_id}/description.json", "w") as fw:
        json.dump(description, fw, indent=4)


if __name__ == "__main__":
    preprocess_hdfs("./HDFS/HDFS_100k.log_structured.csv", "./HDFS/anomaly_label.csv")
