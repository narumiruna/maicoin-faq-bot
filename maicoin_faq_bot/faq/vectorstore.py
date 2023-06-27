from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from loguru import logger

from .docs import FAQDocuments


def load_faq_vectorstore(json_file: str = 'maicoin_faq_zh.json'):
    logger.info(f'loading json file: {json_file}')
    docs = FAQDocuments.from_json(json_file)

    logger.info('splitting documents...')
    docs = RecursiveCharacterTextSplitter(chunk_size=3000).split_documents(docs)

    logger.info('creating vectorstore...')
    vectorstore = FAISS.from_documents(docs, embedding=OpenAIEmbeddings())

    return vectorstore
