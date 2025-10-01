# 🌐 BrowserArena

A live open-web agent evaluation platform that collects user-submitted tasks and user preferences.

## News
- [2025/10] We are releasing our evaluation platform code for BrowserArena.

## Installation

### Prerequisites

- PostgreSQL
- Rust
- CMake
- uv

### Quick Start

**Mac:**
```bash
# Install system dependencies and start PostgreSQL service
brew install postgresql rust cmake pkg-config icu4c
brew services start postgresql

# Set up environment variables for icu4c (add these to ~/.zshrc or ~/.bashrc for persistence)
export PATH="/opt/homebrew/opt/icu4c/bin:/opt/homebrew/opt/icu4c/sbin:${PATH}"
export PKG_CONFIG_PATH="/opt/homebrew/opt/icu4c/lib/pkgconfig:${PKG_CONFIG_PATH}"
unset CC CXX
echo 'export PATH="/opt/homebrew/opt/icu4c/bin:/opt/homebrew/opt/icu4c/sbin:${PATH}"' >> ~/.zshrc
echo 'export PKG_CONFIG_PATH="/opt/homebrew/opt/icu4c/lib/pkgconfig:${PKG_CONFIG_PATH}"' >> ~/.zshrc

# Install UV (if not already installed), then restart terminal or run: source ~/.zshrc (or ~/.bashrc)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync

mkdir -p logs

# Install Playwright browser
uv run playwright install chromium
```

**Ubuntu:**
```bash
# Install system dependencies
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install postgresql rustc cmake pkg-config libicu-dev

# Start and enable PostgreSQL service
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Install UV (if not already installed), then restart terminal or run: source ~/.bashrc
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install all Python dependencies with uv sync (this installs FastChat, browser-use, and all requirements)
# Note: Large downloads (torch, etc.) may require UV_HTTP_TIMEOUT=300
uv sync

# Create required directories
mkdir -p logs

# Install Playwright dependencies and browser
uv run playwright install-deps
uv run playwright install chromium
```

### Troubleshooting

**Timeout during large downloads:**
```bash
UV_HTTP_TIMEOUT=300 uv sync
```

**Python runtime error (pyo3_runtime.PanicException):**
```bash
uv pip install pyopenssl cryptography --upgrade
```

## Execute BrowserArena

First, in `FastChat/api_endpoint.json`, add the OpenRouter Models you want to evaluate on:

```
{
    "meta-llama/llama-4-maverick:free": {
        "model_name": "meta-llama/llama-4-maverick:free",
        "api_base": "",
        "api_type": "",
        "api_key": "",
        "anony_only": false
    }
}
```

Then, export your OpenRouter API Key as follows:

```
export OPENROUTER_API_KEY=""
```

In a terminal, run the following two commands in two separate windows (from root dir):
```bash
uv run python -m fastchat.serve.controller
```

```bash
uv run python -m fastchat.serve.gradio_web_server_multi --register-api-endpoint-file FastChat/api_endpoint.json
```

For headless rendering:
```bash
xvfb-run -a uv run python -m fastchat.serve.gradio_web_server_multi --register-api-endpoint-file FastChat/api_endpoint.json
```

## Compute Leaderboard
```bash
uv run python FastChat/fastchat/serve/monitor/clean_battle_data.py
```

This generates a battles file at `clean_battle_<date>.json` in the FastChat directory.

```bash
uv run python FastChat/fastchat/serve/monitor/elo_analysis.py --clean-battle-file FastChat/clean_battle_<date>.json
```

## Acknowledgements
Our codebase is derived from the [FastChat](https://github.com/lm-sys/FastChat) and [BrowserUse](https://github.com/browser-use/browser-use) codebases. Please also cite the following works if you find this repository helpful.

```
@misc{zheng2023judging,
      title={Judging LLM-as-a-judge with MT-Bench and Chatbot Arena},
      author={Lianmin Zheng and Wei-Lin Chiang and Ying Sheng and Siyuan Zhuang and Zhanghao Wu and Yonghao Zhuang and Zi Lin and Zhuohan Li and Dacheng Li and Eric. P Xing and Hao Zhang and Joseph E. Gonzalez and Ion Stoica},
      year={2023},
      eprint={2306.05685},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

```
@software{browser_use2024,
  author = {Müller, Magnus and Žunič, Gregor},
  title = {Browser Use: Enable AI to control your browser},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/browser-use/browser-use}
}
```