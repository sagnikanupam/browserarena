#!/usr/bin/env python3
import re
import os

# GIF mapping from the extraction
gif_mapping = {
    1: {'agent1': '12_05_2025_03_54_21_KhV', 'agent2': '12_05_2025_03_54_21_dPl'},
    2: {'agent1': '12_05_2025_04_05_21_mCX', 'agent2': '12_05_2025_04_05_21_eYC'},
    3: {'agent1': '12_05_2025_03_56_29_aty', 'agent2': '12_05_2025_03_56_29_nGd'},
    4: {'agent1': '12_05_2025_05_13_04_KAL', 'agent2': '12_05_2025_05_13_04_yaD'},
    5: {'agent1': '07_05_2025_15_10_56_EiU', 'agent2': '07_05_2025_15_10_56_hQu'},
    6: {'agent1': '09_05_2025_19_57_33_HgY', 'agent2': '09_05_2025_19_57_33_CFM'},
    7: {'agent1': '09_05_2025_20_13_59_XFw', 'agent2': '09_05_2025_20_13_59_wLl'},
    8: {'agent1': '09_05_2025_20_11_09_SCY', 'agent2': '09_05_2025_20_11_09_FrD'},
    9: {'agent1': '09_05_2025_20_58_23_Idv', 'agent2': '09_05_2025_20_58_23_unx'},
    10: {'agent1': '09_05_2025_21_30_52_Eso', 'agent2': '09_05_2025_21_30_52_wcO'},
    11: {'agent1': '09_05_2025_21_21_06_MSm', 'agent2': '09_05_2025_21_21_06_kAA'},
    12: {'agent1': '09_05_2025_21_28_12_OnA', 'agent2': '09_05_2025_21_28_12_RJD'},
    13: {'agent1': '09_05_2025_21_40_30_XjR', 'agent2': '09_05_2025_21_40_30_YEs'},
    14: {'agent1': '09_05_2025_21_53_08_BHP', 'agent2': '09_05_2025_21_53_08_Zvg'},
    15: {'agent1': '09_05_2025_21_53_08_BHP', 'agent2': '09_05_2025_21_53_08_Zvg'},
    16: {'agent1': '09_05_2025_21_36_59_crp', 'agent2': '09_05_2025_21_36_59_AvQ'},
    17: {'agent1': '09_05_2025_21_41_14_YwM', 'agent2': '09_05_2025_21_41_14_DvE'},
    18: {'agent1': '09_05_2025_22_23_29_cgB', 'agent2': '09_05_2025_22_23_29_IXB'},
    19: {'agent1': '09_05_2025_22_44_26_YNT', 'agent2': '09_05_2025_22_44_26_xEk'},
    20: {'agent1': '09_05_2025_22_39_53_FuI', 'agent2': '09_05_2025_22_39_53_htb'},
    21: {'agent1': '09_05_2025_23_09_20_BUj', 'agent2': '09_05_2025_23_09_20_MfC'},
    22: {'agent1': '10_05_2025_00_00_27_IAX', 'agent2': '10_05_2025_00_00_27_TVe'},
    23: {'agent1': '10_05_2025_01_14_08_PHv', 'agent2': '10_05_2025_01_14_08_RLX'},
    24: {'agent1': '10_05_2025_01_25_04_TAY', 'agent2': '10_05_2025_01_25_04_TTJ'},
    25: {'agent1': '10_05_2025_01_37_05_xIM', 'agent2': '10_05_2025_01_37_05_jxR'}
}

def embed_gif_in_html(interaction_num, file_path):
    """Embed GIF directly in HTML after Agent header"""
    
    # Read the HTML file
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Get GIF IDs for this interaction
    agent1_gif = gif_mapping[interaction_num]['agent1']
    agent2_gif = gif_mapping[interaction_num]['agent2']
    
    # Create GIF URLs
    agent1_url = f"https://davisrbr.github.io/verified-gifs-only/gifs/{agent1_gif}.gif"
    agent2_url = f"https://davisrbr.github.io/verified-gifs-only/gifs/{agent2_gif}.gif"
    
    # First, remove any existing GIF divs (from previous script)
    # Pattern to match the link div we added before
    link_pattern = r'<div style="padding: 10px; background: #f0f0f0; text-align: center;">\s*<a href="[^"]+\.gif"[^>]+>View Agent [12] GIF →</a>\s*</div>'
    content = re.sub(link_pattern, '', content)
    
    # Create embedded GIF HTML
    agent1_embed = f'''<div style="padding: 15px; background: #f8f9fa; text-align: center;">
                    <img src="{agent1_url}" alt="Agent 1 Browser Interaction" style="max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px;">
                </div>'''
    
    agent2_embed = f'''<div style="padding: 15px; background: #f8f9fa; text-align: center;">
                    <img src="{agent2_url}" alt="Agent 2 Browser Interaction" style="max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px;">
                </div>'''
    
    # Find and replace Agent 1 section
    agent1_pattern = r'(<div class="agent-header">Agent 1</div>)'
    agent1_replacement = f'\\1\n                {agent1_embed}'
    content = re.sub(agent1_pattern, agent1_replacement, content)
    
    # Find and replace Agent 2 section
    agent2_pattern = r'(<div class="agent-header">Agent 2</div>)'
    agent2_replacement = f'\\1\n                {agent2_embed}'
    content = re.sub(agent2_pattern, agent2_replacement, content)
    
    # Write the updated content back
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"Embedded GIFs in interaction {interaction_num}: {file_path}")

# Process all HTML files
html_dir = 'survey_html_snippets'
for i in range(1, 26):
    file_name = f'interaction_{i:02d}.html'
    file_path = os.path.join(html_dir, file_name)
    
    if os.path.exists(file_path):
        embed_gif_in_html(i, file_path)
    else:
        print(f"Warning: {file_path} not found")

print("\nAll HTML files have been updated with embedded GIFs!")