import os
import pandas as pd
import requests
import json
import time
import ast
from openai import OpenAI
from typing import Callable, Dict, Any

client = OpenAI(
   )


def identityFunc(v: str):
    return v


def parseToListFunc(v: str):
    value_list = []
    if isinstance(v, list):
        return v
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


def makeBatchRequest_OpenAI(
    df: pd.DataFrame,
    newColumnName: str,
    jobBodyFunc: Callable[[pd.Series], Dict[str, Any]],
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
