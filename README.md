# BrowserArena

## Installation

### Prerequisites

PostGreSQL, Rust, Cmake

For Mac:
```
brew install postgresql (or equivalent installation on Linux)
brew install rust cmake
brew reinstall pkg-config icu4c
ls /opt/homebrew/opt/icu4c/bin
ls /opt/homebrew/opt/icu4c/sbin
export PATH="/opt/homebrew/opt/icu4c/bin:/opt/homebrew/opt/icu4c/sbin:${PATH}"
export PKG_CONFIG_PATH="/opt/homebrew/opt/icu4c/lib/pkgconfig:${PKG_CONFIG_PATH}"
unset CC CXX
```

```
cd FastChat
python3.11 -m pip install --upgrade pip  # enable PEP 660 support
python3.11 -m pip install -e ".[model_worker,webui]"
cd ..
Python3.11 -m pip install numpy==2.2.4
python3.11 -m pip install -e browser-use
playwright install chromium
python3.11 -m pip install polyglot pyicu pycld2
````

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
```
python3.11 -m fastchat.serve.controller
````

```
python3.11 -m fastchat.serve.gradio_web_server_multi --register-api-endpoint-file api_endpoint.json
```

## Compute Leaderboard
```
python3.11 fastchat/serve/monitor/clean_battle_data.py
```

This generates a battles file at clean_battle_<date>.json in the FastChat directory. 

```
cd FastChat
python3.11 fastchat/serve/monitor/elo_analysis.py --clean-battle-file clean_battle_<date>.json
```