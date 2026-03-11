#!/usr/bin/env python3

import re
import os
import shutil

def extract_survey_interactions():
    # Read the verified interactions file
    with open('/Users/davisbrown/browserarena/extracted_verified_interactions_fixed.txt', 'r') as f:
        content = f.read()
    
    # Split into individual interactions
    interactions = content.split('INTERACTION ')[1:]  # Skip the header
    
    # Find interactions where both models have GIFs (not N/A)
    complete_interactions = []
    
    for interaction in interactions:
        lines = interaction.strip().split('\n')
        
        # Extract interaction number
        interaction_num = lines[0].strip()
        
        # Check if both model attempts have GIFs
        gif1_match = re.search(r'MODEL ATTEMPT 1:.*\n.*\n.*GIF ID: (.+)', interaction)
        gif2_match = re.search(r'MODEL ATTEMPT 2:.*\n.*\n.*GIF ID: (.+)', interaction)
        
        if gif1_match and gif2_match:
            gif1 = gif1_match.group(1).strip()
            gif2 = gif2_match.group(1).strip()
            
            # Make sure neither is N/A
            if gif1 != 'N/A' and gif2 != 'N/A':
                complete_interactions.append({
                    'num': interaction_num,
                    'content': 'INTERACTION ' + interaction,
                    'gif1': gif1,
                    'gif2': gif2
                })
    
    print(f"Found {len(complete_interactions)} interactions with complete GIFs")
    
    # Select first 25 interactions
    survey_interactions = complete_interactions[:25]
    
    # Create survey interactions file
    output_lines = []
    output_lines.append("=== BROWSERARENA SURVEY INTERACTIONS ===")
    output_lines.append("Total interactions: 25 (subset with all GIFs available)")
    output_lines.append("=" * 50)
    output_lines.append("")
    
    all_gifs = []
    
    for i, interaction in enumerate(survey_interactions):
        output_lines.append(interaction['content'])
        all_gifs.append(interaction['gif1'])
        all_gifs.append(interaction['gif2'])
    
    # Write survey interactions file
    with open('/Users/davisbrown/browserarena/survey_interactions.txt', 'w') as f:
        f.write('\n'.join(output_lines))
    
    print(f"Created survey_interactions.txt with {len(survey_interactions)} interactions")
    print(f"Total GIFs needed: {len(all_gifs)}")
    
    return all_gifs

if __name__ == "__main__":
    gifs_to_copy = extract_survey_interactions()
    
    # Save GIF list for later use
    with open('/Users/davisbrown/browserarena/survey_gifs_list.txt', 'w') as f:
        for gif in gifs_to_copy:
            f.write(gif + '\n')
    
    print(f"Saved list of {len(gifs_to_copy)} GIFs to survey_gifs_list.txt")