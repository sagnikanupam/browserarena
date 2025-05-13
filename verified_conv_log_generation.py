import pandas as pd
import os
import json 
import re


verified_df = pd.read_csv("data/ProlificBrowserArenaGIFFeedbackForm_May 12, 2025_08.32_usable.csv", header=0)
verified_df = verified_df.drop(0, axis=0)
print(verified_df.head())

interactions = []
valid_votes = ["tievote", "leftvote", "rightvote"]
log_response_map = {
    "leftvote": "left",
    "rightvote": "right",
    "tievote": "tie",
    "left": "leftvote",
    "right": "rightvote",
    "tie": "tievote",
}

vote_exceptions = {
    "R_7K2orvMBCopRT1N": "right",
    "R_2V8SR0DJbtPGI1g": "left",
    "R_1HpZ6JVauYl0juX": "tie",
    "R_8TAcNC1yB5ZbaMh": "right",
    "R_8CkohlPVxuxQ0j0": "left",
    "R_5NPRDinMw8qYIr8": "left",
    "R_8KfnqUjZfnE1Pxu": "left"
}

discounted_ids = []

response_id_prompt_map = {}
prompt_response_id_map = {}
prompt_list = []
for index, row in verified_df.iterrows():
    response_id = row["ResponseId"]
    prompt = row["Q2"]
    if response_id in vote_exceptions:
        vote = vote_exceptions[response_id]
    else:
        vote = row["Q41"].split()[0].lower()
        vote = re.sub(r"[^a-zA-Z]", "", vote)
    if vote not in ["left", "right", "tie"]:
        raise ValueError(f"Invalid vote found: {vote} for response ID {response_id}")
    verified_df.at[index, "Vote"] = vote.lower()
    if response_id not in response_id_prompt_map:
        response_id_prompt_map[response_id] = {"Prompt": prompt, "Vote": vote, "left_logid": row["Q7"], "right_logid": row["Q25"], "IP": row["IPAddress"], "text_submissions": []}
        prompt_list.append(prompt.strip())
    else:
        raise ValueError(f"Duplicate response ID found: {response_id}")
    if prompt not in prompt_response_id_map:
        prompt_response_id_map[prompt.strip()] = {"ResponseID": [response_id], "Presence" : 0, "Count": 1}
    else:
        print(f"Duplicate prompt found: {prompt.strip()}")
        prompt_response_id_map[prompt.strip()]["ResponseID"].append(response_id)
        prompt_response_id_map[prompt.strip()]["Count"] += 1
        
for file in os.listdir("FastChat/conv_logs"):
    if file.endswith(".json"):
        with open(os.path.join("FastChat/conv_logs", file), "r") as f:
            for line in f:
                data = json.loads(line)
                if data["type"].lower()=="chat":
                    message_strs = [x[1] for x in data["state"]["messages"]]
                    for message_str in message_strs:
                        strp_message_str = str(message_str).strip()
                        if strp_message_str in prompt_list:
                            response_id = prompt_response_id_map[strp_message_str]["ResponseID"][0]
                            response_id_prompt_map[response_id]["text_submissions"].append(data)
                if data["type"].lower() in valid_votes:
                    message_strs = [x[1] for x in data["states"][0]["messages"]]
                    for message_str in message_strs:
                        strp_message_str = str(message_str).strip()
                        if strp_message_str in prompt_list:
                            prompt_response_id_map[strp_message_str]["Presence"] += 1
                            response_id = prompt_response_id_map[strp_message_str]["ResponseID"][0]
                            if log_response_map[data["type"].lower()] != response_id_prompt_map[response_id]["Vote"]: 
                                print(f"Vote mismatch for {response_id}: {log_response_map[data['type'].lower()]} vs {response_id_prompt_map[response_id]['Vote']}")
                                discounted_ids.append(response_id)
                            else:
                                for i in range(len(data["states"])):
                                    str_messages = []
                                    for pair in data["states"][i]["messages"]:
                                        str_messages.append([str(x) for x in pair])
                                    data["states"][i]["messages"] = str_messages
                                interactions.append(data)


print(f"Number of prompts voted on: {len(interactions)} out of {len(prompt_list)}")

num_fixed = 0
for prompt in prompt_response_id_map.keys():
    template = {
        "tstamp": 0.0, 
        'type': None,
        'models': ['', ''],
        'states': [],
        'ip': "",
    }
    response_id = prompt_response_id_map[prompt]["ResponseID"][0]
    presence = prompt_response_id_map[prompt]["Presence"]
    count = prompt_response_id_map[prompt]["Count"]
    left_logid = response_id_prompt_map[response_id]["left_logid"]
    right_logid = response_id_prompt_map[response_id]["right_logid"]
    if presence != count:
        print(f"Prompt {prompt} has {presence} presence and {count} count")
    if presence == 0:
        template["type"] = log_response_map[response_id_prompt_map[response_id]["Vote"]]
        with open(os.path.join("FastChat/prompts_and_outputs", f"{left_logid}.json"), "r") as f:
            left_data = json.load(f)
        with open(os.path.join("FastChat/prompts_and_outputs", f"{right_logid}.json"), "r") as f:
            right_data = json.load(f)
        template["ip"] = response_id_prompt_map[response_id]["IP"]
        for submission in response_id_prompt_map[response_id]["text_submissions"]:
            if left_data["model_name"] == submission["model"]:
                template["states"].append(submission["state"])
        if len(template["states"]) != 1: 
            print(f"Left submission not found for {response_id}")
            template["states"].append(
                {
                "template_name": "",
                "system_message": "",
                "roles": ["Human", "Assistant"], 
                "messages": [["Human", str(left_data["prompt"][0]["content"])], ["Assistant", str(left_data["output"])]],
                "offset": 0, 
                "conv_id": "", 
                "model_name": left_data["model_name"]
                }
            )
        for submission in response_id_prompt_map[response_id]["text_submissions"]:
            if right_data["model_name"] == submission["model"]:
                template["states"].append(submission["state"])
                template["tstamp"] = float(submission["tstamp"])
        if len(template["states"]) != 2:
            print(f"Right submission not found for {response_id}")
            template["states"].append(
                {
                "template_name": "",
                "system_message": "",
                "roles": ["Human", "Assistant"], 
                "messages": [["Human", str(right_data["prompt"][0]["content"])], ["Assistant", str(right_data["output"])]],
                "offset": 0, 
                "conv_id": "", 
                "model_name": right_data["model_name"]
                }
            )
        interactions.append(template)
        num_fixed += 1    

for interaction in interactions:
    for i in range(len(interaction["states"])):
        for msg_arr in interaction["states"][i]["messages"]:
            for pair_element in msg_arr:
                if not pair_element:
                    print(f"Empty message found in interaction {interaction['tstamp']} with {msg_arr}")

print(f"Number of fixed interactions: {num_fixed}, and total interactions: {len(interactions)}")
print(len([x for x in interactions if x["type"].lower() == "tievote"]))
print(len([x for x in interactions if x["type"].lower() == "leftvote"]))
print(len([x for x in interactions if x["type"].lower() == "rightvote"]))

with open("FastChat/verified_conv_log/verified_interactions-conv.json", "w") as f:
    for interaction in interactions:
        f.write(json.dumps(interaction) + "\n")
        