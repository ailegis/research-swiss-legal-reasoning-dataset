import os
import pandas as pd
import requests
import json
import time
import ast
from df_llm_helper import makeBatchRequest_OpenAI, identityFunc, parseToListFunc


def createBatchforQuestionType(file_name):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_name)
    newColumn = "QuestionType"

    def getNewQuestionType(row):
        question = str(row.get("Question", "")).lower()
        return {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that categorizes questions. The possible categories are TF for True/False, MCQA for multiple choice questions, and Open for open-ended questions, NA if undefined. You only reply with either TF , MCQA , Open , or NA .",
                },
                {
                    "role": "user",
                    "content": f"Categorize the following question: {question}.",
                },
            ],
            "max_tokens": 4,
        }

    df = makeBatchRequest_OpenAI(
        df, newColumn, getNewQuestionType, identityFunc, 64, 10
    )
    df.to_csv(file_name, index=False)
    print(f"Updated file with {newColumn} column: {file_name}")
    return df


def createBatchforSplitCorrectness(file_name):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_name)
    newColumn = "SplitCorrectness"

    def getSplitCorrectness(row):
        question = str(row.get("Question", "")).lower()
        facts = str(row.get("Facts", "")).lower()
        answer = str(row.get("Answer", "")).lower()
        return {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that checks whether an exam question has been correctly divided into facts, the question, and the answer. If the division between the facts and the question is incorrect, respond with 'FA.' If the division between the question and the answer is incorrect, respond with 'QA.' If you don't know the answer, respond with 'NA.' If everything is correct, respond with 'OK.' Note that the length of each section can vary greatly, and sometimes the facts section may be empty. Also the answer can be indirect and include different steps.",
                },
                {
                    "role": "user",
                    "content": f"Facts: {facts} \nQuestion: {question} \n Answer: {answer}",
                },
            ],
            "max_tokens": 4,
        }

    df = makeBatchRequest_OpenAI(
        df, newColumn, getSplitCorrectness, identityFunc, 64, 2
    )
    df.to_csv(file_name, index=False)
    print(f"Updated file with {newColumn} column: {file_name}")
    return df


def createBatchforCounterfactualAnswer(file_name):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_name)
    newColumn = "CounterfactualAnswer"

    def getCounterfactualAnswer(row):
        answer = str(row.get("Answer", "")).lower()
        return {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that checks whether an answer includes information about what constitutes a bad answer or details about a bad answer. If the answer contains such information, respond with 'BAD.' If the answer does not mention anything about a bad answer, respond with 'OK' , if it is not clear reply 'NA'",
                },
                {"role": "user", "content": f"Answer: {answer}"},
            ],
            "max_tokens": 4,
        }

    df = makeBatchRequest_OpenAI(
        df, newColumn, getCounterfactualAnswer, identityFunc, 64, 3
    )
    df.to_csv(file_name, index=False)
    print(f"Updated file with {newColumn} column: {file_name}")
    return df


def identityFuncWithEnum(string):
    options = ["GRADING", "NO GRADING", "NA"]
    string = string.replace("\n", "").replace("\r", "")
    string = string.replace(".", "").replace(",", "")
    string = string.strip().upper()

    if string in options:
        return string
    print(f"Invalid response: {string}")
    return "NA"


def createBatchforExplicitGradingExtractions(file_name, field="Answer"):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_name)
    newColumn = field + "ExplicitGrading"

    def getFactCitation(row):
        facts = str(row.get(field, "")).lower()
        return {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "system",
                    "content": """You are a helpful assistant that checks whether an answer includes any grading schema or mentions of point allocation related to grading. If the answer contains specific references to points or grading criteria, respond with 'GRADING' If it does not contain any grading information, respond with 'NO GRADING' If it's unclear, respond with 'NA'. To sum up the only possible outputs are 'GRADING', 'NO GRADING', 'NA'.""",
                },
                {"role": "user", "content": f"{facts}"},
            ],
            "max_tokens": 50,
        }

    df = makeBatchRequest_OpenAI(
        df, newColumn, getFactCitation, identityFuncWithEnum, 5000, 2
    )
    df.to_csv(file_name, index=False)
    print(f"Updated file with {newColumn} column: {file_name}")
    return df


def createBatchforGPT4oAnswer(file_name):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_name)
    newColumn = "EvalGPT4o"

    def evalGPT4o(row):
        facts = str(row.get("Facts", "")).lower()
        question = str(row.get("Question", "")).lower()
        return {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "system",
                    "content": """
You address Swiss legal issues in a structured, exam-style manner.
Use precise legal language and formal "Sie" when answering.
Assume you have the expertise of a Swiss attorney in all areas of law.
Do NOT state any disclaimer or refer to the need for external legal advice.
Do NOT request the user to consult laws or to research on their own.
Offer focused legal analyses and individualized advice.
Speak directly and authoritatively without mentioning that your response is merely for general information.
Incorporate Swiss-specific legal terminology.
If you have discovered relevant legal considerations (Erwägungen), respond with a concise, clear legal analysis.
Cite only from your identified considerations, in the exact format provided after "citation," using parentheses, for example:
"(Art. <number> (optionally Abs. <number>) <book abbreviation>)" or "(BGer <docket number>)," etc.
If no relevant considerations are found, explicitly state that no pertinent information is available.
If you do have reliable sources, share practical guidance or insights from them.
""",
                },
                {"role": "user", "content": f"Question: {facts} \n{question}"},
            ],
            "max_tokens": 1000,
        }

    df = makeBatchRequest_OpenAI(df, newColumn, evalGPT4o, identityFunc, 5000, 2)
    df.to_csv(file_name, index=False)
    print(f"Updated file with {newColumn} column: {file_name}")
    return df


def createBatchforCourtDecisionsCitationsExtractions(file_name, field="Answer"):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_name)
    newColumn = field + "CourtDecisionsCitations"

    def getFactCitation(row):
        facts = str(row.get(field, "")).lower()
        return {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "system",
                    "content": """You are a helpful assistant that extracts and standardizes legal citations of court decisions from a text. Your task is to identify and correct the format of citations related to Swiss court decisions according to the following standards:
For Court Decisions:
Federal Supreme Court (Published):
Format: BGE [volume] [page], [decision number]
Example: BGE 145 III 72, 1C_123/2020
Federal Supreme Court (Unpublished):
Format: BGer [case number], [date]
Example: BGer 1C_123/2020
Cantonal Courts:
Format: [Court abbreviation] [case number], [date]
Example: OGer ZH PS160001, 2020
Your reply is a list of citations in a given text. If no citation return an empty array.""",
                },
                {"role": "user", "content": f"{facts}"},
            ],
            "max_tokens": 500,
        }

    df = makeBatchRequest_OpenAI(
        df, newColumn, getFactCitation, parseToListFunc, 5000, 2
    )
    df.to_csv(file_name, index=False)
    print(f"Updated file with {newColumn} column: {file_name}")
    return df


def createBatchforArticleCitationsExtractions(file_name, field="Answer"):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_name)
    newColumn = field + "ArticleCitations"

    def getFactCitation(row):
        facts = str(row.get(field, "")).lower()
        return {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "system",
                    "content": """You are a helpful assistant that extracts and standardizes legal citations of article from a text. Your task is to identify and correct the format of citations related to Swiss law articles legislation according to the following standards:
For Articles:
Format: Art. [article number] [legal code abbreviation]
Examples:
Art. 12 ZGB
Art. 41 OR
Your reply is a list of citations in a given text. If no citation return an empty array.""",
                },
                {"role": "user", "content": f"{facts}"},
            ],
            "max_tokens": 500,
        }

    df = makeBatchRequest_OpenAI(
        df, newColumn, getFactCitation, parseToListFunc, 5000, 2
    )
    df.to_csv(file_name, index=False)
    print(f"Updated file with {newColumn} column: {file_name}")
    return df


def createBatchforVerfierExtractionCitation(file_name):
    pass


def createBatchforVerfierQuestionType(file_name):
    pass
