#!/usr/bin/env python3

import os

def update_index_html():
    # Get list of GIFs in the directory
    gif_dir = '/Users/davisbrown/browserarena/verified-gifs-only/gifs'
    gif_files = sorted([f for f in os.listdir(gif_dir) if f.endswith('.gif')])
    
    # Generate JavaScript array
    gif_array = ',\n            '.join([f'"{gif}"' for gif in gif_files])
    
    html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BrowserArena Survey Interaction GIFs</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .header { background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .stats { display: flex; gap: 20px; margin: 20px 0; }
        .stat { background: #007acc; color: white; padding: 15px; border-radius: 6px; text-align: center; }
        .gif-container { margin: 20px 0; padding: 20px; border: 1px solid #ddd; background: white; border-radius: 8px; }
        .gif-id { font-weight: bold; color: #007acc; font-size: 18px; margin-bottom: 10px; }
        .gif-image { max-width: 100%; height: auto; border-radius: 4px; }
        .metadata { background: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 4px; }
        .copy-btn { background: #28a745; color: white; border: none; padding: 8px 12px; border-radius: 4px; cursor: pointer; margin-left: 10px; }
        .copy-btn:hover { background: #218838; }
        .interaction-header { background: #e7f3ff; padding: 10px; margin: 20px 0; border-radius: 4px; font-weight: bold; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🤖 BrowserArena Survey Interaction GIFs</h1>
        <p>Curated subset of 25 interactions for survey purposes - each interaction shows two models attempting the same task.</p>
        
        <div class="stats">
            <div class="stat">
                <div style="font-size: 24px; font-weight: bold;">50</div>
                <div>Survey GIFs</div>
            </div>
            <div class="stat">
                <div style="font-size: 24px; font-weight: bold;">25</div>
                <div>Interactions</div>
            </div>
            <div class="stat">
                <div style="font-size: 24px; font-weight: bold;">2</div>
                <div>Models per Task</div>
            </div>
        </div>
    </div>

    <div id="gif-list"></div>

    <script>
        // Generate GIF list dynamically
        const gifNames = [
            ''' + gif_array + '''
        ];

        // Function to copy GIF URL to clipboard
        function copyGifUrl(gifName) {
            const url = `https://davisrbr.github.io/verified-gifs-only/gifs/${gifName}`;
            navigator.clipboard.writeText(url).then(() => {
                alert(`Copied: ${url}`);
            });
        }

        // Generate GIF containers grouped by interaction
        function generateGifList() {
            const container = document.getElementById('gif-list');
            
            // Group GIFs by interaction (assuming they come in pairs)
            for (let i = 0; i < gifNames.length; i += 2) {
                const interactionNum = Math.floor(i / 2) + 1;
                
                // Add interaction header
                const header = document.createElement('div');
                header.className = 'interaction-header';
                header.textContent = `Interaction ${interactionNum} of 25`;
                container.appendChild(header);
                
                // Add both GIFs for this interaction
                for (let j = 0; j < 2 && (i + j) < gifNames.length; j++) {
                    const gifName = gifNames[i + j];
                    const gifId = gifName.replace('.gif', '');
                    
                    const gifContainer = document.createElement('div');
                    gifContainer.className = 'gif-container';
                    
                    gifContainer.innerHTML = `
                        <div class="gif-id">
                            Model ${j + 1} - GIF ID: ${gifId}
                            <button class="copy-btn" onclick="copyGifUrl('${gifName}')">Copy URL</button>
                        </div>
                        <div class="metadata">
                            <strong>Direct URL:</strong> https://davisrbr.github.io/verified-gifs-only/gifs/${gifName}<br>
                            <strong>File:</strong> ${gifName}
                        </div>
                        <img src="gifs/${gifName}" alt="${gifId}" class="gif-image" loading="lazy">
                    `;
                    
                    container.appendChild(gifContainer);
                }
            }
        }

        // Generate the list when page loads
        document.addEventListener('DOMContentLoaded', generateGifList);
    </script>
</body>
</html>'''
    
    with open('/Users/davisbrown/browserarena/verified-gifs-only/index.html', 'w') as f:
        f.write(html_content)
    
    print(f"Updated index.html with {len(gif_files)} GIFs")

if __name__ == "__main__":
    update_index_html()