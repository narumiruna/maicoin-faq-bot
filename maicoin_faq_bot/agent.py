import os
from typing import List

from langchain.agents import AgentType
from langchain.agents import initialize_agent
from langchain.base_language import BaseLanguageModel
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.tools import BaseTool
from loguru import logger
from telegram import Update
from telegram.ext import ContextTypes
from telegraph import Telegraph

from maicoin_faq_bot.retriever import MaiCoinFAQRetriever


class MaiCoinFAQAgent:

    def __init__(self, llm: BaseLanguageModel, tools: List[BaseTool]):
        self.llm = llm
        self.tools = tools

        self.agents = {}

    @classmethod
    def from_env(cls):
        model_name = os.environ.get('OPENAI_MODEL_NAME', 'gpt-3.5-turbo-0613')
        llm = ChatOpenAI(model_name=model_name, temperature=0.0)

        faq_file = os.environ.get('MAICOIN_FAQ_FILE', 'maicoin_faq_zh.json')
        tools = [MaiCoinFAQRetriever.from_json(faq_file)]

        return cls(llm=llm, tools=tools)

    def run(self):
        agent = self.create_agent()
        while True:
            try:
                question = input("User: ")
                resp = agent.run(question)
                print('Agent:', resp)
            except KeyboardInterrupt:
                break

    def create_agent(self):
        return initialize_agent(tools=self.tools,
                                llm=self.llm,
                                agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
                                memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True),
                                verbose=True)

    async def chat(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        logger.info('update: {}', update)

        chat_id = update.effective_chat.id
        if chat_id not in self.agents:
            logger.info('create new agent for chat_id: {}', chat_id)
            self.agents[chat_id] = self.create_agent()

        response = self.agents[chat_id].run(update.message.text)
        logger.info('response: {}', response)

        if len(response) > 8000:
            response = short_text(response)

        message = await context.bot.send_message(chat_id=chat_id, text=response)
        logger.info('message: {}', message)


def short_text(content: str):
    telegraph = Telegraph()
    telegraph.create_account(short_name='MaiCoin FAQ Bot')

    response = telegraph.create_page(title='MaiCoin FAQ Bot', html_content=content)
    return response['url']
