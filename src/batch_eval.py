import pandas as pd
import os

from feature_extraction import createBatchforGPT4oAnswer


def load_data_to_df(path):
    """Load CSV data from file or directory path into DataFrame"""

    # Check if path exists
    if not os.path.exists(path):
        raise ValueError(f"Path does not exist: {path}")

    # Handle single file
    if os.path.isfile(path):
        if not path.endswith(".csv"):
            raise ValueError(f"File must be CSV format: {path}")
        df = pd.read_csv(path)

    # Handle directory
    else:
        files = [f for f in os.listdir(path) if f.endswith(".csv")]
        if not files:
            raise ValueError(f"No CSV files found in directory: {path}")

        df = pd.read_csv(os.path.join(path, files[0]))
        for file in files[1:]:
            temp_df = pd.read_csv(os.path.join(path, file))
            df = pd.concat([df, temp_df], ignore_index=True)

    # Display DataFrame information
    print("\nDataFrame Info:")
    print(df.info())
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nBasic statistics:")
    print(df.describe())

    return df


if __name__ == "__main__":

    file_path = "./data/law_exams_dev.csv"
    createBatchforGPT4oAnswer(file_path)

    file_path = "./data/law_exams_test_id.csv"
    createBatchforGPT4oAnswer(file_path)

    file_path = "./data/law_exams_test_ood.csv"
    createBatchforGPT4oAnswer(file_path)
