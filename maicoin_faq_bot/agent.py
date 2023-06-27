import os
from typing import List

from langchain.agents import AgentType
from langchain.agents import initialize_agent
from langchain.base_language import BaseLanguageModel
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.schema import SystemMessage
from langchain.tools import BaseTool
from langchain.tools import Tool
from loguru import logger
from telegram import Update
from telegram.ext import ContextTypes
from telegraph import Telegraph

from .chain import get_faq_chain

system_content = ('你是 MaiCoin 的智慧客服，請你在回答問題時，遵守以下規則：\n'
                  '1. 永遠使用繁體中文\n'
                  '2. 在對話前，你會優先搜尋 MaiCoin FAQ 資料庫取得有用的資訊\n'
                  '3. 要在每一句對話後面加上emoji，種類要多變\n')


class MaiCoinFAQAgent:
    system_message = SystemMessage(content=system_content)

    def __init__(self, llm: BaseLanguageModel, tools: List[BaseTool]):
        self.agent = initialize_agent(tools=tools, llm=llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=True)

    @classmethod
    def from_env(cls):
        model_name = os.environ.get('OPENAI_MODEL_NAME', 'gpt-3.5-turbo-0613')
        llm = ChatOpenAI(model_name=model_name, temperature=0.0)

        tools = [
            Tool.from_function(
                name='MaiCoin-FAQ',
                description=('Useful for when you need to answer questions about MaiCoin, '
                             'MAX exchange or cryptocurrency in general. '
                             'Input should be in the form of a question containing full context'),
                func=get_faq_chain().run,
            )
        ]
        return cls(llm=llm, tools=tools)

    def run(self):
        while True:
            try:
                question = input("User: ")
                resp = self.agent.run([self.system_message, HumanMessage(content=question)])
                print('Agent:', resp)
            except KeyboardInterrupt:
                break

    async def chat(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        logger.info('update: {}', update)

        response = self.agent.run([self.system_message, HumanMessage(content=update.message.text)])

        logger.info('response: {}', response)

        if len(response) > 8000:
            response = short_text(response)

        message = await context.bot.send_message(chat_id=update.effective_chat.id, text=response)
        logger.info('message: {}', message)


def short_text(content: str):
    telegraph = Telegraph()
    telegraph.create_account(short_name='MaiCoin FAQ Bot')

    response = telegraph.create_page(title='MaiCoin FAQ Bot', html_content=content)
    return response['url']
