import os
import pandas as pd

if __name__ == "__main__":
    file_dir = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(f"{file_dir}/preprocessed/HDFS.log_templates.csv")
    padding = pd.DataFrame({"EventId": ["E0"], "EventTemplate": ["padding"]})
    df = pd.concat([padding, df]).reset_index(drop=True)
    json_data = df.to_json(f"{file_dir}/templates.json", orient="records")
