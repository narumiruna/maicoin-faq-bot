from langchain.agents import AgentType
from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from loguru import logger
from telegram import Update
from telegram.ext import ContextTypes
from telegraph import Telegraph

from maicoin_faq_bot.retriever import MaiCoinFAQRetriever


class MaiCoinFAQAgent:

    def __init__(self, model_name='gpt-3.5-turbo-0613', faq_file: str = 'maicoin_faq_zh.json'):
        self.llm = ChatOpenAI(model_name=model_name)
        self.tools = [MaiCoinFAQRetriever.from_json(faq_file)]

        self.agents = {}

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

        agent_resp = self.agents[chat_id].run(update.message.text)
        logger.info('agent response: {}', agent_resp)

        if len(agent_resp) > 9500:
            agent_resp = short_text(agent_resp)

        bot_resp = await context.bot.send_message(chat_id=chat_id, text=agent_resp)
        logger.info('bot response: {}', bot_resp)


def short_text(content: str):
    telegraph = Telegraph()
    telegraph.create_account(short_name='MaiCoin FAQ Bot')

    response = telegraph.create_page(title='MaiCoin FAQ Bot', content=content)
    return response['url']
