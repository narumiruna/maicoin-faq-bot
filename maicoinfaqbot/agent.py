from langchain.agents import AgentType
from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from loguru import logger
from telegram import Update
from telegram.ext import ContextTypes

from maicoinfaqbot.retriever import MaiCoinFAQRetriever


class MaiCoinFAQAgent:

    def __init__(self, model_name='gpt-3.5-turbo-0613', faq_file: str = 'data/maicoin_faq_zh.json'):
        llm = ChatOpenAI(model_name=model_name)
        tools = [MaiCoinFAQRetriever.from_json(faq_file)]
        memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
        self.agent = initialize_agent(tools=tools,
                                      llm=llm,
                                      agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
                                      memory=memory,
                                      verbose=True)

    async def chat(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        logger.info('update: {}', update)

        agent_resp = self.agent.run(update.message.text.rstrip('/' + self.chat_command))
        logger.info('agent response: {}', agent_resp)

        bot_resp = await context.bot.send_message(chat_id=update.effective_chat.id,
                                                  text=agent_resp,
                                                  reply_to_message_id=update.message.id)
        logger.info('bot response: {}', bot_resp)
