from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from loguru import logger

from maicoin_faq_bot.utils import load_json


def main():
    load_dotenv()

    f = 'maicoin_faq_zh.json'
    logger.info(f'Loading json file: {f}')
    data = load_json(f)

    logger.info('splitting documents...')
    docs = RecursiveCharacterTextSplitter().split_documents(
        [Document(page_content=d['body'], metadata={
            'title': d['title'],
            'url': d['url']
        }) for d in data])

    logger.info('creating vectorstore...')
    vectorstore = Chroma.from_documents(docs, embedding=OpenAIEmbeddings())
    print(vectorstore)

    logger.info('creating chain...')
    chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model_name="gpt-3.5-turbo-0613"),
        memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True),
        retriever=vectorstore.as_retriever(),
        verbose=True,
    )

    while True:
        try:
            question = input("User: ")
            resp = chain.run(question)
            print('Agent:', resp)
        except KeyboardInterrupt:
            break


if __name__ == '__main__':
    main()
