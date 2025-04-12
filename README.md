# BrowserArena

## Installation

### Prerequisites

PostGreSQL, Rust, Cmake

For Mac:
```
brew install postgresql (or equivalent installation on Linux)
brew install rust cmake
```

```
cd FastChat
python3.11 -m pip install --upgrade pip  # enable PEP 660 support
python3.11 -m pip install -e ".[model_worker,webui]"
cd ..
Python3.11 -m pip install numpy==2.2.4
python3.11 -m pip install -e browser-use
playwright install chromium
````

## Execute BrowserArena

First, in `FastChat/api_endpoint.json`, add your OpenAI API key:

```
{
    "gpt-4o-2024-05-13": {
        "model_name": "gpt-4o",
        "api_base": "https://api.openai.com/v1",
        "api_type": "openai",
        "api_key": < YOUR OPENAI API KEY >,
        "anony_only": false
    }
}
```

In a terminal, run the following two commands in two separate windows:
```
python3.11 -m fastchat.serve.controller
````

```
python3.11 -m fastchat.serve.gradio_web_server_multi --register-api-endpoint-file api_endpoint.json
```