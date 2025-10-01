# 🌐 BrowserArena

A live open-web agent evaluation platform that collects user-submitted tasks and user preferences.

## News
- [2025/10] We are releasing our evaluation platform code for BrowserArena.

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
cd FastChat
python3.11 -m pip install --upgrade pip  # enable PEP 660 support
python3.11 -m pip install -e ".[model_worker,webui]"
cd ..
python3.11 -m pip install -e browser-use
playwright install chromium
python3.11 -m pip install polyglot pyicu pycld2
```

For Ubuntu:
```
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.11
sudo apt install postgresql rustc cmake python3-pip
cd FastChat
python3.11 -m pip install -e ".[model_worker,webui]" --use-pep517   
cd ..
python3.11 -m pip install -e browser-use --use-pep517
```

Add `export PATH="/usr/bin/python3.11:$PATH"` to `~/.bashrc` and `source ~/.bashrc` and restart terminal.
```
playwright install-deps
playwright install chromium
```

If there is a `pyo3_runtime.PanicException: Python API call failed` error, it can be fixed by `python3.11 -m pip install pyopenssl cryptography --upgrade`.

If there is an error regarding missing `sentence-transformers` library, run `python3.11 -m pip install sentence-transformers`.

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
For headless rendering:
```
xvfb-run -a python3.11 -m fastchat.serve.gradio_web_server_multi --register-api-endpoint-file api_endpoint.json
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

## Acknowledgements
Our codebase is derived from the [FastChat](https://github.com/lm-sys/FastChat) and [BrowserUser](https://github.com/browser-use/browser-use) codebases. Please also cite the following works if you find this repository helpful.

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