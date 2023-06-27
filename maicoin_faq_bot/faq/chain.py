import os

from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferWindowMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from loguru import logger

from .docs import FAQDocuments


def initialize_faq_chain(json_file: str = 'maicoin_faq_zh.json'):
    model_name = os.environ.get('OPENAI_MODEL_NAME', 'gpt-3.5-turbo-0613')
    llm = ChatOpenAI(model_name=model_name)

    logger.info(f'loading json file: {json_file}')
    docs = FAQDocuments.from_json(json_file)

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
