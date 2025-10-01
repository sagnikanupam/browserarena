import json
with open("clean_battle_conv_20250514_global_data_list.json", "r") as f:
    global_data_list = json.load(f)
    
prompt_list = []
for data_list in global_data_list:
    for data in data_list:
        prompt = data["states"][0]["messages"][data["states"][0]["offset"]][1]
        prompt_list.append(prompt)
with open("clean_battle_conv_20250514_global_data_list_prompts.json", "w") as f:
    json.dump(prompt_list, f, indent=4)