import os
import pandas as pd

if __name__ == "__main__":
    file_dir = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(f"{file_dir}/preprocessed/HDFS.log_templates.csv")
    json_data = df.to_json(f"{file_dir}/output.json", orient="records")
