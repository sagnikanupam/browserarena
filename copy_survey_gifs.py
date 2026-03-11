#!/usr/bin/env python3

import os
import shutil

def copy_survey_gifs():
    # Read the list of GIFs to copy
    with open('/Users/davisbrown/browserarena/survey_gifs_list.txt', 'r') as f:
        gif_ids = [line.strip() for line in f if line.strip()]
    
    source_dir = '/Users/davisbrown/browserarena/FastChat/gifs'
    dest_dir = '/Users/davisbrown/browserarena/verified-gifs-only/gifs'
    
    # Ensure destination directory exists
    os.makedirs(dest_dir, exist_ok=True)
    
    copied = 0
    missing = []
    
    for gif_id in gif_ids:
        source_path = os.path.join(source_dir, gif_id)
        dest_path = os.path.join(dest_dir, gif_id)
        
        if os.path.exists(source_path):
            shutil.copy2(source_path, dest_path)
            copied += 1
            print(f"Copied: {gif_id}")
        else:
            missing.append(gif_id)
            print(f"Missing: {gif_id}")
    
    print(f"\nSummary:")
    print(f"- Copied: {copied} GIFs")
    print(f"- Missing: {len(missing)} GIFs")
    
    if missing:
        print("\nMissing GIFs:")
        for gif in missing:
            print(f"  - {gif}")
    
    return copied, missing

if __name__ == "__main__":
    copy_survey_gifs()