from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from loguru import logger

from ..document_loaders import FAQLoader


def load_faq_vectorstore(json_file: str = 'maicoin_faq_zh.json'):
    logger.info(f'loading json file: {json_file}')
    docs = FAQLoader().load_and_split(json_file)

    logger.info('creating vectorstore...')
    vectorstore = FAISS.from_documents(docs, embedding=OpenAIEmbeddings())

    return vectorstore
