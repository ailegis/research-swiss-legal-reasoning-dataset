import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import numpy as np
from litellm import batch_completion_models_all_responses
import os
from df_llm_helper import makeBatchRequest_OpenAI, identityFunc, parseToListFunc


def evaluate_all(file_name: str):
    # run_0_shot(file_name)
    # run_COT_0_shot(file_name)
    compute_metrics_gpt4omini_comp(file_name)
    return


def run_0_shot(file_name: str):
    df = pd.read_csv(file_name)

    newColumn = "OpenAI_gpt4omini_run_0_shot"

    def getNewQuestionType(row):
        question = str(row.get("Question", "")).lower()
        return {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "system",
                    "content": "Du bist ein Experte im Bereich Rechtswissenschaften und bereitest Studierende auf juristische Prüfungen vor.",
                },
                {
                    "role": "user",
                    "content": f"Beantworte diese Frage in deutscher Sprache: {question}.",
                },
            ],
            "max_tokens": 500,
        }

    df = makeBatchRequest_OpenAI(
        df, newColumn, getNewQuestionType, identityFunc, 64, 10
    )
    df.to_csv(file_name, index=False)
    print(f"Updated file with QuestionType column: {file_name}")
    return df


def run_1_shot(file_name: str):
    return


def run_5_shot(file_name: str):
    return


def run_COT_0_shot(file_name: str):
    df = pd.read_csv(file_name)

    newColumn = "OpenAI_gpt4omini_run_COT_0_shot"

    def getNewQuestionType(row):
        question = str(row.get("Question", "")).lower()
        return {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "system",
                    "content": "Du bist ein Experte im Bereich Rechtswissenschaften und bereitest Studierende auf juristische Prüfungen vor. Beantworte die Frage Schritt für Schritt mit einer logischen Herleitung (Chain of Thought), um den Denkprozess verständlich zu machen.",
                },
                {
                    "role": "user",
                    "content": f"Beantworte diese Frage in deutscher Sprache und erkläre deine Herangehensweise: {question}.",
                },
            ],
            "max_tokens": 500,
        }

    df = makeBatchRequest_OpenAI(
        df, newColumn, getNewQuestionType, identityFunc, 64, 10
    )
    df.to_csv(file_name, index=False)
    print(f"Updated file with QuestionType column: {file_name}")
    return df


def compute_metrics_gpt4omini_comp(file_name: str):
    df = pd.read_csv(file_name)

    for col in ["OpenAI_gpt4omini_run_COT_0_shot", "OpenAI_gpt4omini_run_0_shot"]:
        newColumn = "metric_gpt4omini_0_" + col

        def getNewQuestionType(row):
            question = str(row.get("Question", "")).lower()
            real_answer = str(row.get("Answer", "")).lower()
            predicted_answer = str(row.get(col, "")).lower()
            return {
                "model": "gpt-4o-mini",
                "messages": [
                    {
                        "role": "system",
                        "content": "Du bist ein Experte im Bereich Rechtswissenschaften. Deine Aufgabe ist es, Antworten auf juristische Prüfungsfragen zu bewerten. Vergleiche die echte Antwort und die vorhergesagte Antwort für die folgende Frage und bewerte die Genauigkeit der vorhergesagten Antwort auf einer Skala von 0 bis 100, wobei 100 eine perfekte Übereinstimmung darstellt.",
                    },
                    {
                        "role": "user",
                        "content": f"Frage: {question}\nEchte Antwort: {real_answer}\nVorhergesagte Antwort: {predicted_answer}\nWie genau ist die vorhergesagte Antwort im Vergleich zur echten Antwort? Gib eine Bewertung zwischen 0 und 100 ab.",
                    },
                ],
                "max_tokens": 5,
            }

        df = makeBatchRequest_OpenAI(
            df, newColumn, getNewQuestionType, identityFunc, 64, 2
        )
        df.to_csv(file_name, index=False)
        print(f"Updated file with QuestionType column: {file_name}")
    return df


def compute_metrics(file_name: str):
    return
