import os
import openai
import json
import re

SYSTEM_PROMPT = """ You are an LLM judge evaluating the output of an agent attempting to traverse different websites to solve a task. The task involves navigating to a website with a cookie banner or pop-up ad and completing the assigned task. You are provided with the task and the agent's output, and your goal is to identify if the agent was successful in closing the cookie banner or pop-up ad and progressing with the task.

You should output a JSON string in the format {"banner_detected": `bool`, "banner_closed": `bool`} where the "banner_detected" key indicates whether the agent found a cookie banner or pop-up ad at any point while trying to complete the task, and the "banner_closed" key indicates whether the agent successfully closed it. If no cookie banner or pop-up ad was detected, both values should be false.

Make sure that all the keys are present in the output, even if their value is false. Only output the JSON string, do not include any other text, and make sure that the only keys in the JSON string are the ones listed above. Do not include any additional keys or values in the output.
"""

USER_PROMPT = ""

OUTPUT_DIR = "data/systematic_prompts_and_outputs_evals/cookie_banner"
OUTPUT_DIR_TXT = "data/systematic_prompts_and_outputs_evals/cookie_banner/raw_outputs"

def parse_output(content: str) -> str:
    """Parses the last block between ```json or ``` fences from the provided text."""

    # First check for JSON code blocks
    json_match = re.findall(r'```json\n(.*?)```', content, re.DOTALL)
    if json_match:
        # Get the last JSON block
        return json_match[::-1][0].strip()

    # Fallback to generic code blocks
    code_match = re.findall(r'```(.*?)```', content, re.DOTALL)
    if code_match:
        # Return the last code block
        return code_match[::-1][0].strip()

    # If no code blocks found, parse backtick syntax
    backtick_matches = re.findall(r'`(.*?)`\\b', content)
    if backtick_matches:
        return backtick_matches[::-1][0].strip()

    return content.strip()


def make_api_call(user_prompt: str, unique_run_id: str) -> None:
    """Function to make the API call using provided system and user prompts and save the output in a JSON file."""
    try:
        # Checking if the output directory exists, if not making the directory
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
        if not os.path.exists(OUTPUT_DIR_TXT):
            os.makedirs(OUTPUT_DIR_TXT)
        
        # Setting the API key for authentication
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError(
                "An OPENAI_API_KEY environment variable must be set.")

        # Configuring the completion parameters
        completion_params = {
            "model": "o4-mini",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 1.0
        }
        # Making the API call
        response = openai.chat.completions.create(**completion_params)
        print(f"Response received for unique_run_id: {unique_run_id} with response {response}.")
        # Retrieving the output from the response
        llm_output = response.choices[0].message.content
        with open(os.path.join(OUTPUT_DIR_TXT, f"{unique_run_id}_raw_output.txt"), "w") as f:
            f.write(llm_output)

        # Parse model output and validate response
        parsed_output = parse_output(llm_output)
        parsed_output_json = json.loads(parsed_output)
        output_path = os.path.join(OUTPUT_DIR, f"{unique_run_id}.json")
        with open(output_path, "w") as outfile:
            json.dump(parsed_output_json, outfile)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    
    
def calculate_banner_percentages():
    """Calculate the percentage of times each LLM experiences a cookie banner and closes it."""
    import glob
    
    strategy_counts = {}
    total_tasks = {}
    MODEL_NAME_LIST = ["deepseek_deepseek-r1", "meta-llama_llama-4-maverick", "anthropic_claude-3.7-sonnet_thinking", "google_gemini-2.5-pro-preview-03-25", "openai_o4-mini"]
    filenames = {}
    for x in MODEL_NAME_LIST:
        filenames[x] = []
    
    for json_file in glob.iglob(os.path.join(OUTPUT_DIR, "run_*.json")):
        with open(json_file, 'r') as file:
            data = json.load(file)
            llm_name = None
            # Extract the LLM's name from the filename
            for model_name in MODEL_NAME_LIST:
                if json_file.find(f"run_{model_name}") != -1:
                    llm_name = model_name
                    break
            if llm_name is None:
                print(f"Warning: LLM name not found in {json_file}. Skipping this file.")
                continue
            # Initialize LLM in both dictionaries
            if llm_name not in strategy_counts:
                strategy_counts[llm_name] = {k: 0 for k in data.keys()}
                total_tasks[llm_name] = 0
            filenames[llm_name].append(json_file)
            
            if data.get("banner_detected", False):
                total_tasks[llm_name] += 1

            for strategy, value in data.items():
                if value:
                    strategy_counts[llm_name][strategy] = strategy_counts[llm_name].get(strategy, 0) + 1

    # Calculate the percentages
    llm_strategy_percentages = {}
    for llm, strategies in strategy_counts.items():
        llm_strategy_percentages[llm] = {}
        print(f"Total tasks for {llm}: {total_tasks[llm]}.")
        for strategy, count in strategies.items():
            percentage = (count / total_tasks[llm]) * 100
            llm_strategy_percentages[llm].update({strategy: round(percentage, 2)})

    print(f"Strategy percentages: {json.dumps(llm_strategy_percentages, indent=4)}")
    #print(f"Filenames: {json.dumps(filenames, indent=4)}")


def run_llm_as_judge():
    """
    Function to run the LLM as a judge to evaluate the outputs of agents traversing Expedia.

    Raises:
        ValueError: If the OPENAI_API_KEY environment variable is not set.
        FileNotFoundError: If the JSON file paths do not exist.
        Exception: For any other errors during the API call or file operations.
    """
    
    # Paths to JSON files to be processed
    json_filepaths = [
        "FastChat/fastchat/serve/systematic_prompts/cookie_banner_tasks_json_task_id_dict.json"
    ]
    
    #json_filepaths = json_filepaths[:1]  # For testing, only use the first file
    sys_prompt_dir = "FastChat/systematic_prompts_and_outputs/"
    
    for json_filepath in json_filepaths:
        with open(json_filepath, "r") as file:
            json_dict_tasks = json.load(file)

        for model_name, tasks in json_dict_tasks.items():
            if model_name == "x-ai/grok-3-beta":
                continue
            for prompt, unique_run_id in tasks.items():
                # Reading the user_prompt from a text file
                json_path = os.path.join(sys_prompt_dir, f"{unique_run_id}.json")
                try:
                    with open(json_path, "r") as txt_file:
                        data = json.load(txt_file)
                        user_prompt = data.get("output", None)
                        if user_prompt is None:
                            raise ValueError(
                                f"User prompt not found in {json_path}. Please check the file.")
                        # Calling the API function with the extracted user_prompt and unique_run_id
                        make_api_call(user_prompt, unique_run_id)
                except FileNotFoundError:
                    print(f"Error: {json_path} not found. Skipping this task.")
                    continue

if __name__ == "__main__":
    run_llm_as_judge()
    calculate_banner_percentages()