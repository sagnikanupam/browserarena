from langchain_openai import ChatOpenAI
from browser_use import Agent
import asyncio
from dotenv import load_dotenv
load_dotenv()
import os, json
import logging

with open("api_endpoint.json", "r") as f:
    api_endpoint = json.load(f)
    for key in api_endpoint.keys():
        if api_endpoint[key]["api_type"] == "openai":
            os.environ["OPENAI_API_KEY"] = api_endpoint[key]["api_key"]

async def main():
    agent = Agent(
        generate_gif = True,
        task="Compare the price of gpt-4o and DeepSeek-V3",
        llm=ChatOpenAI(model="gpt-4o"),
    )
    await agent.run()

# logger = logging.getLogger(__name__)
# logger.addHandler(fl)
asyncio.run(main())
# logger.removeHandler(fl)
# fl.close()
