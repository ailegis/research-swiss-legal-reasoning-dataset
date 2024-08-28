import os
import pandas as pd
import requests

def load_csv(file_name, google_docs_link):
    # Ensure the directory exists
    folder = os.path.dirname(file_name)
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"Created directory: {folder}")
    if not os.path.exists(file_name):
        file_id = google_docs_link.split('/d/')[1].split('/')[0]
        export_link = f"https://docs.google.com/spreadsheets/d/{file_id}/export?format=csv"
        response = requests.get(export_link)
        with open(file_name, 'wb') as file:
            file.write(response.content)
        print(f"Downloaded and saved as {file_name}")
    else:
        print(f"Loading from local file: {file_name}")
    
    return pd.read_csv(file_name)


def addextractedCitations(file_name):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_name)
    
    # Add columns for citations
    df['QuestionCitations'] = [[] for _ in range(len(df))]
    df['AnswerCitation'] = [[] for _ in range(len(df))]
    df['FactCitation'] = [[] for _ in range(len(df))]
    
    # Save the updated DataFrame back to CSV
    df.to_csv(file_name, index=False)
    print(f"Updated file with citation columns: {file_name}")
    return df
