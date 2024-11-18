from dataset import load_csv
from evaluate import evaluate_all
from feature_extraction import (
    createBatchforCounterfactualAnswer,
    createBatchforQuestionType,
    createBatchforSplitCorrectness,
    createBatchforArticleCitationsExtractions,
    createBatchforCourtDecisionsCitationsExtractions,
    createBatchforExplicitGradingExtractions
)
from report import (
    generate_dataframe_report,
    generate_visual_report,
    generate_visual_report2,
)


csv_file_name = "data/data.csv"
google_docs_url = "https://docs.google.com/spreadsheets/d/1cRsQS6nuqK9zfJYoOmXegLTZYjSs6IYiTyPkPiQIBsY/edit?gid=1421018847#gid=1421018847"
data = load_csv(csv_file_name, google_docs_url)


run_preprocessing = False
if run_preprocessing:
    # data = createBatchforQuestionType(csv_file_name)
    # data = createBatchforSplitCorrectness(csv_file_name)
    # data = createBatchforCounterfactualAnswer(csv_file_name)
    # data = createBatchforAnswerCitationExtractions(csv_file_name)
    # data = createBatchforQuestionCitationsExtractions(csv_file_name)
    # data = createBatchforFactCitationExtractions(csv_file_name)
    for field in ["Answer", "Facts", "Question"]:
        data = createBatchforArticleCitationsExtractions(csv_file_name, field)
        data = createBatchforCourtDecisionsCitationsExtractions(csv_file_name, field)
        data = createBatchforExplicitGradingExtractions(csv_file_name, field)


run_reports = True
if run_reports:
    generate_dataframe_report(data)
    generate_visual_report(data)
    generate_visual_report2(data)


run_evaluations = False
if run_evaluations:
    evaluate_all(csv_file_name)
