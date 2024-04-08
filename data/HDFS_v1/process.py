import os
import json
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict

if __name__ == "__main__":
    file_dir = os.path.dirname(os.path.abspath(__file__))
    trace_file = f"{file_dir}/preprocessed/Event_traces.csv"
    template_file = f"{file_dir}/preprocessed/HDFS.log_templates.csv"
    dump_dir = f"{file_dir}/../../loglizer/data/HDFS_DeepLog"
    shuffle = True
    valid_ratio = 0.001
    test_ratio = 0.99
    os.makedirs(dump_dir, exist_ok=True)

    traces_df = pd.read_csv(trace_file, engine="c", na_filter=False, memory_map=True)
    templates_df = pd.read_csv(template_file, engine="c", na_filter=False, memory_map=True)

    eid2template = {}
    for _, row in templates_df.iterrows():
        event_id = int(row["EventId"][1:])
        eid2template[event_id] = row["EventTemplate"]
    with open(f"{dump_dir}/eid2template.pkl", "wb") as fw:
        pickle.dump(eid2template, fw)

    session_dict = defaultdict(lambda: {"eids": None, "anomaly": 0})
    for _, row in traces_df.iterrows():
        blk_id = row["BlockId"]
        # "[E5,E22,...,E21]" -> [5, 22, ..., 21]
        session_dict[blk_id]["eids"] = [int(s[1:]) for s in row["Features"][1:-1].split(",")]
        session_dict[blk_id]["anomaly"] = 1 if row["Label"] == "Fail" else 0

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
