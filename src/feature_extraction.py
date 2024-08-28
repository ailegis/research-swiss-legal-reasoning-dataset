import os
import pandas as pd
import requests
import json
import time
import ast
from openai import OpenAI

client = OpenAI()


def identityFunc(v: str):
    return v


def parseToListFunc(v: str):
    value_list = []
    try:
        value = v.replace("```", "").replace("json", "")
        parsed_value = ast.literal_eval(value)
        if isinstance(parsed_value, list):
            value_list = parsed_value
        else:
            raise ValueError(f"Parsed value is not a list: {parsed_value}")
    except (ValueError, SyntaxError) as e:
        print(f"Failed to parse value: {value}. Error: {e}")
        value_list = ["error"]
    return value_list


def makeBatchRequest(df: pd.DataFrame,
                     newColumnName: str,
                     jobBodyFunc,
                     valuePostProcessingFunc,
                     batchSize=64,
                     maxBatches=10,
                     ):
    
    df[newColumnName] = None
    batch_ids = []
    for batch_start in range(0, len(df), batchSize):
        batch_end = batch_start + batchSize
        batch_df = df.iloc[batch_start:batch_end]
        batch_file_name = (
            f"src/batchjobs/batchinput_batch_{batch_start//batchSize + 1}.jsonl"
        )
        with open(batch_file_name, "w") as batch_file:
            for i, row in batch_df.iterrows():
                batch_job = {
                    "custom_id": f"request-{i}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": jobBodyFunc(row),
                }
                batch_file.write(json.dumps(batch_job) + "\n")
        batch_input_file = client.files.create(
            file=open(batch_file_name, "rb"), purpose="batch"
        )
        batch_input_file_id = batch_input_file.id
        b = client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": f"eval job batch {batch_start//batchSize + 1}"},
        )
        print(f"Batch {batch_start//batchSize + 1} created: {b.id}")
        batch_ids.append(b.id)
        if len(batch_ids) >= maxBatches or batch_end >= len(df):
            print(batch_ids)
            completed_batches = []
            while len(batch_ids) > 0:
                time.sleep(10)
                for batch_id in batch_ids.copy():
                    r = client.batches.retrieve(batch_id)
                    # TODO : implement failed scenarios
                    if r.status == "completed":
                        completed_batches.append(r)
                        batch_ids.remove(batch_id)
                    else:
                        print(f"Batch ID: {batch_id}, Status: {r.status}")
            for idx, b in enumerate(completed_batches):
                print(b)
                result_file_id = b.output_file_id
                print(result_file_id)
                result = client.files.content(result_file_id).content
                result_file_name = f"src/batchjobs/completed/batch_{idx}.jsonl"
                with open(result_file_name, "wb") as file:
                    file.write(result)
                results = []
                with open(result_file_name, "r") as file:
                    for line in file:
                        # Parsing the JSON string into a dict and appending to the list of results
                        json_object = json.loads(line.strip())
                        results.append(json_object)
                for res in results:
                    task_id = res["custom_id"]
                    index = int(task_id.split("-")[-1])
                    value = res["response"]["body"]["choices"][0]["message"]["content"]
                    print(f"v:{value}, idx:{index}")
                    df.at[index, newColumnName] = valuePostProcessingFunc(value)
            batch_ids = []
    return df


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
    

    df = makeBatchRequest(df, newColumn, getNewQuestionType, identityFunc, 64 ,10)       
    df.to_csv(file_name, index=False)
    print(f"Updated file with QuestionType column: {file_name}")
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
    

    df = makeBatchRequest(df, newColumn, getSplitCorrectness, identityFunc, 64, 2)       
    df.to_csv(file_name, index=False)
    print(f"Updated file with SplitCorrectness column: {file_name}")
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
    

    df = makeBatchRequest(df, newColumn, getCounterfactualAnswer, identityFunc, 64, 5)       
    df.to_csv(file_name, index=False)
    print(f"Updated file with CounterfactualAnswer column: {file_name}")
    return df


def createBatchforAnswerCitationExtractions(file_name):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_name)
    newColumn = "AnswerCitation"


    def getAnswerCitation(row):
        answer = str(row.get("Answer", "")).lower()
        return {
                        "model": "gpt-4o-mini",
                        "messages": [
                            {
                                "role": "system",
                                "content": """You are a helpful assistant that extracts and standardizes legal citations from a text. Your task is to identify and correct the format of citations related to Swiss law articles and court decisions according to the following standards:
For Articles:
Format: Art. [article number] [legal code abbreviation]
Examples:
Art. 12 ZGB
Art. 41 OR
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
                            {"role": "user", "content": f"{answer}"},
                        ],
                        "max_tokens": 500,
                    }


    df = makeBatchRequest(df, newColumn, getAnswerCitation, parseToListFunc, 64, 3)       
    df.to_csv(file_name, index=False)
    print(f"Updated file with AnswerCitation column: {file_name}")
    return df


def createBatchforQuestionCitationsExtractions(file_name):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_name)
    newColumn = "QuestionCitations"


    def getQuestionCitations(row):
        question = str(row.get("Question", "")).lower()
        return {
                        "model": "gpt-4o-mini",
                        "messages": [
                            {
                                "role": "system",
                                "content": """You are a helpful assistant that extracts and standardizes legal citations from a text. Your task is to identify and correct the format of citations related to Swiss law articles and court decisions according to the following standards:
For Articles:
Format: Art. [article number] [legal code abbreviation]
Examples:
Art. 12 ZGB
Art. 41 OR
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
                            {"role": "user", "content": f"{question}"},
                        ],
                        "max_tokens": 500,
                    }


    df = makeBatchRequest(df, newColumn, getQuestionCitations, parseToListFunc, 64, 3)       
    df.to_csv(file_name, index=False)
    print(f"Updated file with QuestionCitations column: {file_name}")
    return df


def createBatchforFactCitationExtractions(file_name):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_name)
    newColumn = "FactCitation"


    def getFactCitation(row):
        facts = str(row.get("Facts", "")).lower()
        return {
                        "model": "gpt-4o-mini",
                        "messages": [
                            {
                                "role": "system",
                                "content": """You are a helpful assistant that extracts and standardizes legal citations from a text. Your task is to identify and correct the format of citations related to Swiss law articles and court decisions according to the following standards:
For Articles:
Format: Art. [article number] [legal code abbreviation]
Examples:
Art. 12 ZGB
Art. 41 OR
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


    df = makeBatchRequest(df, newColumn, getFactCitation, parseToListFunc, 64, 3)       
    df.to_csv(file_name, index=False)
    print(f"Updated file with FactCitation column: {file_name}")
    return df


def createBatchforVerfierExtractionCitation(file_name):
    pass


def createBatchforVerfierQuestionType(file_name):
    pass

