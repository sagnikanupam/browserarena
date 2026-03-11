import csv
import json

# Read the faithfulness data
mappings = []
with open('/Users/davisbrown/browserarena/agreement-results/data_updated/faithfulness_data.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['Q7'] and '_' in row['Q7'] and row['Q24'] and '_' in row['Q24']:
            mapping = {
                'response_id': row['ResponseId'],
                'task_question': row['Q2'],
                'left_case_id': row['Q7'],
                'right_case_id': row['Q24']
            }
            mappings.append(mapping)

# Save as JSON
with open('/Users/davisbrown/browserarena/agreement-results/survey_case_id_mapping.json', 'w') as f:
    json.dump(mappings, f, indent=2)

# Save as CSV  
with open('/Users/davisbrown/browserarena/agreement-results/survey_case_id_mapping.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['response_id', 'task_question', 'left_case_id', 'right_case_id'])
    writer.writeheader()
    writer.writerows(mappings)

print(f"Found {len(mappings)} survey responses with case ID mappings")
print("\nFirst 5 mappings:")
for m in mappings[:5]:
    print(f"Response: {m['response_id']}")
    print(f"Task: {m['task_question'][:50]}...")
    print(f"Left Case ID: {m['left_case_id']}")
    print(f"Right Case ID: {m['right_case_id']}")
    print("---")
