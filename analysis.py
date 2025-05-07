import pandas as pd
import re
import json

banned_id_list = ["A", "67e1152fe716c33597b9184a", "67dbb1fef3c51a422079a857", "67f7bb9aaa1cb67247023347", "67d11644421446068e1fec14", "67da6c21bcd71cf297d8db10", "67e1152fe716c33597b9184a", "668914213741d9029705d8b1", "6813ee94347bb13e5862b369"]

def get_log_id(log_id_str: str) -> str:
    """
    Extracts the log ID from the given string.

    Args:
        log_id_str (str): The string containing the log ID.

    Returns:
        str: The extracted log ID.
    """
    log_id_str = re.sub('[^A-Za-z0-9_]+', '', log_id_str)
    if "logid" in log_id_str.lower():
        log_id_str = re.split("logid", log_id_str, flags=re.IGNORECASE)[-1].strip()
    return log_id_str

def max_steps(file_path: str) -> int:
    """
    Reads a JSON file and returns the number of valid steps.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    output = data["output"]
    potential_steps = []
    intermediate_strings = output.split("Step ")
    for x in intermediate_strings:
        try:
            potential_steps.append(int(re.sub("[^0-9]", "", x[:2])))
        except Exception as e:
            continue
    return max(potential_steps, default=-1)

def analyze_data(file_path: str, banned_ids: list[str] = banned_id_list):
    """
    Analyzes the JSONL file and prints basic statistics.

    Args:
        file_path (str): CSV file path.
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    df = df[df["DistributionChannel"] != "preview"]
    df = df[df["Q1"].str.strip().str.lower().isin(banned_ids) == False]
    for index, row in df.iterrows():
        correct_left_data_entry = False
        correct_right_data_entry = False
        num_valid_steps_left = 0
        num_valid_steps_right = 0
        max_steps_left = 0
        max_steps_right = 0
        if "Yes" in row["Q4"]:
            if "Yes" in row["Q5"]:
                for i in range(15):
                    if str(row[f"Q{i+8}"]).lower().find("n/a") == -1 and str(row[f"Q{i+8}"]) != "" and str(row[f"Q{i+8}"]) != "nan":
                        num_valid_steps_left += 1
                    else:
                        break
                try:
                    max_steps_left = max_steps(f"FastChat/prompts_and_outputs/{get_log_id(str(row['Q7']))}.json")
                    if num_valid_steps_left == max_steps_left:
                        correct_left_data_entry = True
                except Exception as e:
                    print(f"Error processing left data entry for index {index}: {row['Q7']} with log ID {get_log_id(str(row['Q7']))} with error {e}")
            if "Yes" in row["Q23"]:
                for i in range(15):
                    if str(row[f"Q{i+26}"]).lower().find("n/a") == -1 and str(row[f"Q{i+26}"]) != "" and str(row[f"Q{i+26}"]) != "nan":
                        num_valid_steps_right += 1
                    else:
                        break
                try:
                    max_steps_right = max_steps(f"FastChat/prompts_and_outputs/{get_log_id(str(row['Q25']))}.json")
                    if num_valid_steps_right == max_steps_right:
                        correct_right_data_entry = True
                except Exception as e:
                    print(f"Error processing right data entry for index {index}: {row['Q25']} with log ID {get_log_id(str(row['Q25']))} with error {e}") 
        
        # Set correct data entry status in the DataFrame
        df.at[index, "num_valid_steps_left"] = num_valid_steps_left
        df.at[index, "num_valid_steps_right"] = num_valid_steps_right
        df.at[index, "max_steps_left"] = max_steps_left
        df.at[index, "max_steps_right"] = max_steps_right
        df.at[index, "CorrectLeftDataEntry"] = correct_left_data_entry
        df.at[index, "CorrectRightDataEntry"] = correct_right_data_entry
        df.at[index, "SoftFlag"] = not (correct_left_data_entry and correct_right_data_entry)
        df.at[index, "Flagged"] = not (correct_left_data_entry and correct_right_data_entry) and (max_steps_left > num_valid_steps_left or max_steps_right > num_valid_steps_right)
        
    # Save the updated DataFrame to a new CSV file by suffixing "analyzed" to the original filename
    output_file_path = file_path.replace(".csv", "_analyzed.csv")
    df.to_csv(output_file_path, index=False)
    
    # Display the first few rows of the DataFrame
    print("First few rows of the dataset:")
    print(df.head())

    # Display basic statistics about the dataset
    print("\nBasic statistics:")
    print(df.describe(include='all'))

if __name__ == "__main__":
    # Specify the path to your JSONL file
    file_path = "data/pilot_dataset.csv"
    
    # Call the function to analyze the data
    analyze_data(file_path)