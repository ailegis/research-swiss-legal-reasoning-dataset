import pandas as pd
import os
import re
import matplotlib.pyplot as plt

from feature_extraction import createBatchforGPT4oGrading


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
    
    # Compute total score
    total_score = df["ScoresGPT4o"].sum()
    print(f"\nTotal Score: {total_score}")
    
    # Compute statistics per course
    if "Course" in df.columns:
        course_stats = df.groupby("Course")["ScoresGPT4o"].describe()
        print("\nStatistics per Course:")
        print(course_stats)
        
        # Sort by mean score
        course_stats = course_stats.sort_values(by="mean", ascending=False)
        
        # Plot mean and std per course
        plt.figure(figsize=(16, 8))
        ax = course_stats[['mean', 'std']].plot(kind='bar', figsize=(16, 8), title=f'Mean and Std per Course {path}')
        plt.ylabel("Score")
        plt.xlabel("Course")
        plt.legend(["Mean", "Std Dev"])
        plt.xticks(rotation=90, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Save plot to file with path-specific name
        plot_path = path.replace(".csv", "_scores_plot.png")
        plt.savefig(plot_path, bbox_inches='tight')
        print(f"\nPlot saved to {plot_path}")
        plt.close()

    return df


def extract_scores_from_text(df):
    """Extract numerical scores from the 'GradingGPT4o' text column and store in 'ScoresGPT4o'."""

    def parse_score(text):
        match = re.search(r"The correctness score:\s*\[\[(\d*\.?\d+)\]\]", text)
        if match:
            return float(match.group(1))  # Extract and convert to float
        return None  # Return None if no match found

    df["ScoresGPT4o"] = df["GradingGPT4o"].apply(parse_score)
    return df


if __name__ == "__main__":

    file_paths = [
        "./data/law_exams_dev.csv",
        "./data/law_exams_test_id.csv",
        "./data/law_exams_test_ood.csv",
    ]

    for file_path in file_paths:
        df = load_data_to_df(file_path)
        df = extract_scores_from_text(df)
        df.to_csv(file_path, index=False)
