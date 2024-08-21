from dataset import load_csv
from report import generate_dataframe_report, generate_visual_report, generate_visual_report2


csv_file_name = 'data/data.csv'
google_docs_url = 'https://docs.google.com/spreadsheets/d/1cRsQS6nuqK9zfJYoOmXegLTZYjSs6IYiTyPkPiQIBsY/edit?gid=1470754638#gid=1470754638'
data = load_csv(csv_file_name, google_docs_url)


generate_dataframe_report(data)
generate_visual_report(data)
generate_visual_report2(data)
