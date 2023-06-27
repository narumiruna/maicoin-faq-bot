import os

from langchain.base_language import BaseLanguageModel
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import BaseRetriever
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from loguru import logger
from telegram import Update
from telegram.ext import ContextTypes

from maicoin_faq_bot.utils import load_json


def get_faq_chain():
    model_name = os.environ.get('OPENAI_MODEL_NAME', 'gpt-3.5-turbo-0613')
    llm = ChatOpenAI(model_name=model_name)

    json_file = os.environ.get('MAICOIN_FAQ_JSON', 'maicoin_faq_zh.json')
    logger.info(f'loading json file: {json_file}')
    data = load_json(json_file)

    logger.info('creating documents...')
    docs = []
    for d in data:
        page_content = (f'Title: {d["title"]}\n'
                        f'URL: {d["url"]}\n'
                        f'{d["body"]}')
        docs.append(Document(page_content=page_content))

    logger.info('splitting documents...')
    docs = RecursiveCharacterTextSplitter(chunk_size=3000).split_documents(docs)

    logger.info('creating vectorstore...')
    vectorstore = Chroma.from_documents(docs, embedding=OpenAIEmbeddings())

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        memory=ConversationBufferWindowMemory(memory_key='chat_history', return_messages=True),
        retriever=vectorstore.as_retriever(),
        verbose=True,
    )

    return chain


class MaiCoinFAQChain:

    def __init__(self, llm: BaseLanguageModel, retriever: BaseRetriever):
        self.llm = llm
        self.retriever = retriever

        self.chains = {}

    @classmethod
    def from_env(cls):
        model_name = os.environ.get('OPENAI_MODEL_NAME', 'gpt-3.5-turbo-0613')
        llm = ChatOpenAI(model_name=model_name)

        json_file = os.environ.get('MAICOIN_FAQ_JSON', 'maicoin_faq_zh.json')
        logger.info(f'loading json file: {json_file}')
        data = load_json(json_file)

        logger.info('creating documents...')
        docs = []
        for d in data:
            page_content = (f'Title: {d["title"]}\n'
                            f'URL: {d["url"]}\n'
                            f'{d["body"]}')
            docs.append(Document(page_content=page_content))

        logger.info('splitting documents...')
        docs = RecursiveCharacterTextSplitter().split_documents(docs)

        logger.info('creating vectorstore...')
        vectorstore = Chroma.from_documents(docs, embedding=OpenAIEmbeddings())

        return cls(llm=llm, retriever=vectorstore.as_retriever())

    def create_chain(self):
        logger.info('creating memory...')
        memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

        logger.info('creating chain...')
        chain = ConversationalRetrievalChain.from_llm(llm=self.llm,
                                                      memory=memory,
                                                      retriever=self.retriever,
                                                      verbose=True)

        return chain

    def run(self):
        chain = self.create_chain()
        while True:
            try:
                question = input("User: ")
                resp = chain.run(question)
                print('Agent:', resp)
            except KeyboardInterrupt:
                break

    async def chat(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        logger.info('update: {}', update)

        chat_id = update.effective_chat.id
        if chat_id not in self.chains:
            logger.info('create new chain for chat_id: {}', chat_id)
            self.chains[chat_id] = self.create_chain()

        response = self.chains[chat_id].run(update.message.text)
        logger.info('response: {}', response)

        message = await context.bot.send_message(chat_id=chat_id, text=response)
        logger.info('send message: {}', message)
