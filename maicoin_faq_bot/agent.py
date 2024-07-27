import os
from typing import List

from langchain.agents import AgentType
from langchain.agents import initialize_agent
from langchain.base_language import BaseLanguageModel
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import SystemMessage
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI
from loguru import logger
from telegram import Update
from telegram.ext import ContextTypes
from telegraph import Telegraph

from .tools import MAXTicker
from .tools import create_faq_tool

SYSTEM_CONTENT = """你是 MaiCoin 的智慧客服
永遠使用繁體中文
會使用 faq_search 取得 MaiCoin FAQ 的資訊
在每一句對話後面加上emoji，種類要多變
不知道時就說不知道，不要亂猜
"""


class MaiCoinFAQAgent:
    def __init__(self, llm: BaseLanguageModel, tools: List[BaseTool]):
        self.agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.OPENAI_FUNCTIONS,
            agent_kwargs={"system_message": SystemMessage(content=SYSTEM_CONTENT)},
            memory=ConversationBufferWindowMemory(memory_key="chat_history"),
            verbose=True,
        )

    @classmethod
    def from_env(cls):
        model_name = os.environ.get("OPENAI_MODEL_NAME", "gpt-4o-mini")
        llm = ChatOpenAI(model_name=model_name, temperature=0.0)

        tools = [
            create_faq_tool(llm=llm, json_file="maicoin_faq_zh.json"),
            MAXTicker(),
        ]
        return cls(llm=llm, tools=tools)

    def run(self):
        while True:
            try:
                question = input("User: ")
                resp = self.agent.run(question)
                print("Agent:", resp)
            except KeyboardInterrupt:
                break

    async def chat(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        logger.info("update: {}", update)

        response = self.agent.run(update.message.text)

        logger.info("response: {}", response)

        print("response:", response)
        if len(response) > 1000:
            response = short_text(response)

        message = await context.bot.send_message(chat_id=update.effective_chat.id, text=response)
        logger.info("message: {}", message)


def short_text(content: str):
    telegraph = Telegraph()
    telegraph.create_account(short_name="MaiCoin FAQ Bot")

    response = telegraph.create_page(title="MaiCoin FAQ Bot", html_content=content)
    return response["url"]
