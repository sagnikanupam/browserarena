import pandas as pd
import os
import json
user_data = pd.read_csv('ProlificBrowserArenaGIFFeedbackForm_May 14, 2025_22.32_usable.csv')
user_data["left_model"] = ["Not Found"] * len(user_data)
user_data["right_model"] = ["Not Found"] * len(user_data)
user_data["left_log_id"] = ["Not Found"] * len(user_data)
user_data["right_log_id"] = ["Not Found"] * len(user_data)
dir_list = os.listdir("../FastChat/prompts_and_outputs")
assert len(dir_list) > 0, "No logs found in the prompts_and_outputs directory."
for index, row in user_data.iterrows():
    response_id = row["ResponseId"]
    left_log_id = row["Q7"]
    right_log_id = row["Q25"]
    if f"{left_log_id}.json" in dir_list:
        try:
            with open(f"../FastChat/prompts_and_outputs/{left_log_id}.json", "r") as f:
                left_log = json.load(f)
                print(f"Reading left log for response ID {response_id} with left log id {left_log_id}")
                #Set row's left_model value to left_log["model_name"]
                user_data.loc[index, "left_model"] = left_log["model_name"]
                user_data.loc[index, "left_log_id"] = left_log_id
        except Exception as e:
            print(f"Error reading left log for response ID {response_id} with left log id {left_log_id}: {e}")
            row["left_model"] = "Not Found"
    else:
        print(f"Left log {left_log_id} not found.")
        row["left_model"] = "Not Found"
    if f"{right_log_id}.json" in dir_list:
        try:
            with open(f"../FastChat/prompts_and_outputs/{right_log_id}.json", "r") as f:
                right_log = json.load(f)
                print(f"Reading right log for response ID {response_id} with right log id {right_log_id}")
                user_data.loc[index, "right_model"] = right_log["model_name"]
                user_data.loc[index, "right_log_id"] = right_log_id
        except Exception as e:
            print(f"Error reading right log for response ID {response_id} with right log id {right_log_id}: {e}")
            row["right_model"] = "Not Found"
    else:
        row["right_model"] = "Not Found"

user_data.to_csv("model_feedback_txt/grouped.csv", index=False)
model_list = ["deepseek/deepseek-r1", "meta-llama/llama-4-maverick", "anthropic/claude-3.7-sonnet:thinking", "x-ai/grok-3-beta", "google/gemini-2.5-pro-preview-03-25", "openai/o4-mini"]
#Divide dataframe based on models used on left and right responses
left_model_split = {}
right_model_split = {}
for model in model_list:
    left_model_split[model] = user_data[user_data["left_model"] == model]
    right_model_split[model] = user_data[user_data["right_model"] == model]
    
model_str_dict = {}
for model in model_list:
    model_str = [f"{model}\n"]
    for index, row in left_model_split[model].iterrows():
        model_str += [f"Task Prompt: {row['Q2']} \n"]
        if user_data.loc[index, "left_log_id"] == "Not Found":
            model_str += [f"Agent Trace: Not Available\n"]
        else:
            model_str += [f"Agent Trace: https://github.com/sagnikanupam/browserarena/blob/main/FastChat/prompts_and_outputs/{row['left_log_id']}.json\n"]
        model_str += [f"User Stepwise Feedback: \n"]
        for step in range(8, 23):
            feedback = row[f"Q{step}"]
            if pd.notna(feedback) and str(feedback).lower() != "n/a" and str(feedback).lower() != "na" and str(feedback).lower() != "n\a":
                model_str += [f"Step {step-7} feedback: {feedback}\n"]
        model_str += ["\n\n"]
    
    for index, row in right_model_split[model].iterrows():
        model_str += [f"Task Prompt: {row['Q2']} \n"]
        if user_data.loc[index, "right_log_id"] == "Not Found":
            model_str += [f"Agent Trace: Not Available\n"]
        else:
            model_str += [f"Agent Trace: https://github.com/sagnikanupam/browserarena/blob/main/FastChat/prompts_and_outputs/{row['right_log_id']}.json\n"]
        model_str += [f"User Stepwise Feedback: \n"]
        for step in range(26, 41):
            feedback = row[f"Q{step}"]
            if pd.notna(feedback) and str(feedback).lower() != "n/a" and str(feedback).lower() != "na" and str(feedback).lower() != "n\a":
                model_str += [f"Step {step-25} feedback: {feedback}\n"]
        model_str += ["\n\n"]
            
    model_str += ["\n\n"]
    model_str_dict[model] = " ".join(model_str)
    model_sanitized = model.replace("/", "_").replace(":", "_").replace("-", "_")
    with open(f"model_feedback_txt/{model_sanitized}.txt", "w") as f:
        f.write(model_str_dict[model])