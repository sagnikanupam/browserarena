#!/usr/bin/env python3
"""
Verify that all GIF files referenced in the interactions exist
"""

import json
import os
import re
from typing import Set

def extract_gif_id(message_content: str) -> str:
    """Extract GIF ID from message content using regex"""
    gif_pattern = r'<img src="/gradio_api/file=gifs/([^"]+\.gif)"'
    match = re.search(gif_pattern, message_content)
    return match.group(1) if match else "NOT_FOUND"

def main():
    """Main function to verify GIF files exist"""
    input_file = "/Users/davisbrown/browserarena/FastChat/verified_conv_log/verified_interactions-clean.json"
    gif_dir = "/Users/davisbrown/browserarena/FastChat/gifs"
    
    print("Loading JSON file...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract all GIF IDs
    gif_ids = set()
    
    for i, interaction in enumerate(data, 1):
        states = interaction.get("states", [])
        
        for state in states:
            # Find the last assistant message
            for message in reversed(state.get("messages", [])):
                if isinstance(message, list) and len(message) >= 2:
                    role = message[0]
                    content = message[1]
                    if role.lower() in ["assistant", "model"]:
                        gif_id = extract_gif_id(content)
                        if gif_id != "NOT_FOUND":
                            gif_ids.add(gif_id)
                        break
    
    print(f"Found {len(gif_ids)} unique GIF IDs referenced in interactions")
    
    # Check if GIF files exist
    missing_gifs = []
    existing_gifs = []
    
    for gif_id in gif_ids:
        gif_path = os.path.join(gif_dir, gif_id)
        if os.path.exists(gif_path):
            existing_gifs.append(gif_id)
        else:
            missing_gifs.append(gif_id)
    
    # Check for GIF files that exist but aren't referenced
    if os.path.exists(gif_dir):
        actual_gif_files = set([f for f in os.listdir(gif_dir) if f.endswith('.gif')])
        unreferenced_gifs = actual_gif_files - gif_ids
    else:
        actual_gif_files = set()
        unreferenced_gifs = set()
    
    print(f"\nGIF FILE VERIFICATION RESULTS:")
    print(f"================================")
    print(f"Referenced GIF IDs: {len(gif_ids)}")
    print(f"Existing GIF files: {len(existing_gifs)}")
    print(f"Missing GIF files: {len(missing_gifs)}")
    print(f"Unreferenced GIF files: {len(unreferenced_gifs)}")
    print(f"Total GIF files on disk: {len(actual_gif_files)}")
    
    if missing_gifs:
        print(f"\nMISSING GIF FILES:")
        for gif in sorted(missing_gifs):
            print(f"  - {gif}")
    
    if unreferenced_gifs:
        print(f"\nUNREFERENCED GIF FILES (exist but not in interactions):")
        for gif in sorted(list(unreferenced_gifs))[:10]:  # Show first 10
            print(f"  - {gif}")
        if len(unreferenced_gifs) > 10:
            print(f"  ... and {len(unreferenced_gifs) - 10} more")

if __name__ == "__main__":
    main()