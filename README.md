# 🌐 BrowserArena

A live open-web agent evaluation platform that collects user-submitted tasks and user preferences.

## News
- [2025/10] We are releasing our evaluation platform code for BrowserArena.

## Installation

### Prerequisites

PostGreSQL, Rust, Cmake, UV (Python package manager)

For Mac:
```bash
# Install system dependencies
brew install postgresql rust cmake pkg-config icu4c

# Set up environment variables for icu4c
export PATH="/opt/homebrew/opt/icu4c/bin:/opt/homebrew/opt/icu4c/sbin:${PATH}"
export PKG_CONFIG_PATH="/opt/homebrew/opt/icu4c/lib/pkgconfig:${PKG_CONFIG_PATH}"
unset CC CXX

# Install UV (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Python 3.11 and create virtual environment
uv python install 3.11
uv venv --python 3.11

# Install FastChat
cd FastChat
uv pip install -e ".[model_worker,webui]" --python ../.venv/bin/python
cd ..

# Install browser-use and additional dependencies
uv pip install -e browser-use --python .venv/bin/python
uv pip install polyglot pyicu pycld2 --python .venv/bin/python

# Install Playwright browser
uv run playwright install chromium
```

For Ubuntu:
```bash
# Install system dependencies
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install postgresql rustc cmake pkg-config libicu-dev

# Install UV (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Python 3.11 and create virtual environment
uv python install 3.11
uv venv --python 3.11

# Install FastChat
cd FastChat
uv pip install -e ".[model_worker,webui]" --python ../.venv/bin/python
cd ..

# Install browser-use and additional dependencies
uv pip install -e browser-use --python .venv/bin/python
uv pip install polyglot pyicu pycld2 --python .venv/bin/python

# Install Playwright dependencies and browser
uv run playwright install-deps
uv run playwright install chromium
```

**Troubleshooting:**

If there is a `pyo3_runtime.PanicException: Python API call failed` error, it can be fixed by running:
```bash
uv pip install pyopenssl cryptography --upgrade --python .venv/bin/python
```

If there is an error regarding missing `sentence-transformers` library, run:
```bash
uv pip install sentence-transformers --python .venv/bin/python
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

In a terminal, run the following two commands in two separate windows:
```bash
uv run python -m fastchat.serve.controller
```

```bash
uv run python -m fastchat.serve.gradio_web_server_multi --register-api-endpoint-file api_endpoint.json
```

For headless rendering:
```bash
xvfb-run -a uv run python -m fastchat.serve.gradio_web_server_multi --register-api-endpoint-file api_endpoint.json
```

## Compute Leaderboard
```bash
uv run python fastchat/serve/monitor/clean_battle_data.py
```

This generates a battles file at clean_battle_<date>.json in the FastChat directory.

```bash
cd FastChat
uv run python fastchat/serve/monitor/elo_analysis.py --clean-battle-file clean_battle_<date>.json
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