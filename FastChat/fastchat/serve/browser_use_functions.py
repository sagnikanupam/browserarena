from browser_use import Agent
from browser_use.browser.browser import Browser, BrowserConfig
from browser_use.browser.context import BrowserContext
import asyncio
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

import os
from typing import Optional

from dotenv import load_dotenv
from langchain_core.utils.utils import secret_from_env
from langchain_openai import ChatOpenAI
from pydantic import Field, SecretStr

load_dotenv()

class ChatOpenRouter(ChatOpenAI):
    openai_api_key: Optional[SecretStr] = Field(
        alias="api_key", default_factory=secret_from_env("OPENROUTER_API_KEY", default=None)
    )
    @property
    def lc_secrets(self) -> dict[str, str]:
        return {"openai_api_key": "OPENROUTER_API_KEY"}

    def __init__(self,
                 openai_api_key: Optional[str] = None,
                 **kwargs):
        openai_api_key = os.environ.get("OPENROUTER_API_KEY") # openai_api_key or os.environ.get("OPENROUTER_API_KEY")
        super().__init__(base_url="https://openrouter.ai/api/v1", openai_api_key=openai_api_key, **kwargs)

async def call_browser(task_prompt: str = "Compare the price of gpt-4o and DeepSeek-V3", model="deepseek/deepseek-r1:free", unique_run_index: str = "abc123", anonymous: bool = False):
    agent = Agent(
        generate_gif = True,
        tool_calling_method = "raw",
        task = task_prompt,
        llm=ChatOpenRouter(model_name=model),
        unique_run_index = unique_run_index,
        max_steps = 15,
        conversion = True if "deepseek" in model else False,
        anonymous=anonymous,  
    )
    #headless_param browser_context=BrowserContext( browser=Browser(config=BrowserConfig(headless=True)),), 
    await agent.run()