import asyncio
import argparse
import json
from datetime import datetime
from pathlib import Path
import os 

from fastchat.serve.browser_use_functions import call_browser

global_task_name_dict_filepath = None
global_task_name_dict = {} if global_task_name_dict_filepath is None else json.load(open(global_task_name_dict_filepath, "r"))
BATCH_SIZE = 1

async def run_tasks(config_data: list):
    # Parse JSON and collect tasks
    tasks_to_run = []
    for entry_index, entry in enumerate(config_data):
        model = entry['model']
        global_task_name_dict.setdefault(model, {})
        model_sanitized = model.replace("/", "_").replace(":", "_")  # Sanitize model name for use in filenames
        prompt_index = entry['prompt_index']
        for prompt in entry['prompts']:
            # Generate unique timestamp and identifier
            run_index = f"run_{model_sanitized}_{prompt_index}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            global_task_name_dict[model][prompt] = run_index

            # Ensure output directory exists
            create_path = Path.cwd().absolute() / "gifs"
            create_path.mkdir(parents=True, exist_ok=True)      
            
            tasks_to_run.append(call_browser(task_prompt=prompt, model=model, unique_run_index=run_index, anonymous=False))
    
    # Await all tasks concurrently
    await asyncio.gather(*tasks_to_run, return_exceptions=True)

def create_logs_json():
    for model, tasks in global_task_name_dict.items():
        for prompt, run_index in tasks.items():
            with open(f"logs/{run_index}.txt", "r") as f:
                text_output = f.read()
                image = '\n' + f'<img src="/gradio_api/file=gifs/{run_index}.gif" alt="GIF" />'
                with open(f"systematic_prompts_and_outputs/{run_index}.json", "w") as f:
                    json.dump({"prompt": prompt, "output": text_output, "image": image, "unique_run_index": f"{run_index}", "model_name": model}, f)

def main(args):
    # Use the async function in an async event loop
    with open(args.config_path, 'r') as file:
        config_data_all = json.load(file)    
    config_data_split = [config_data_all[i:i + BATCH_SIZE] for i in range(0, len(config_data_all), BATCH_SIZE)]
    directory = os.path.dirname(args.config_path)
    filename = os.path.basename(args.config_path)
    for config_data in config_data_split:
        print(f"Running {len(config_data)} tasks in this batch...")
        print(f"Config data: {config_data}")
        asyncio.run(run_tasks(config_data))
        with open(os.path.join(directory, f"{filename.replace('.', '_')}_task_id_dict.json"), "w") as f:
            json.dump(global_task_name_dict, f, indent=4)
    create_logs_json()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run browser automation tasks from a JSON configuration.")
    parser.add_argument("config_path", type=str, help="Path to JSON config file defining prompt-model pairings.")
    # Add an optional flag to enable anonymity
    parser.add_argument("--anonymize", action="store_true", help="Enable anonymous mode for tasks.")
    args = parser.parse_args()
    main(args)
