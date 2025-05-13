import os
import json
current_directory = os.getcwd()
for file in os.listdir(current_directory):
    try:
        with open(file, "r") as f:
            result_dict = json.load(f)
            if result_dict["model_name"] == "x-ai/grok-3-beta":
                print(file)
    except Exception as e:
        continue