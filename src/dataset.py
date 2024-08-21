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