from __future__ import annotations

from langchain.base_language import BaseLanguageModel
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.tools import BaseTool
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from loguru import logger

from ..document_loaders import FAQLoader


def create_faq_tool(
    llm: BaseLanguageModel, json_file: str = "maicoin_faq_zh.json", max_output_chars: int = 4000
) -> FAQTool:
    max_output_chars = max_output_chars
    vectorstore = load_faq_vectorstore(json_file=json_file)
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True),
        verbose=True,
    )
    return FAQTool(max_output_chars=max_output_chars, vectorstore=vectorstore, chain=chain)


class FAQTool(BaseTool):
    name = "faq_search"
    description: str = (
        "Useful for when you need to answer questions."
        "You should ALWAYS use this."
        "Input should be a fully formed question."
    )
    max_output_chars: int = 4000
    vectorstore: FAISS
    chain: ConversationalRetrievalChain

    class Config:
        arbitrary_types_allowed = True

    def _run(self, query: str) -> str:
        res = self.chain.invoke(query)
        res = res.get("answer", "I'm sorry, I don't know how to answer that.")
        return res[: self.max_output_chars]


def load_faq_vectorstore(json_file: str = "maicoin_faq_zh.json"):
    logger.info(f"loading json file: {json_file}")
    docs = FAQLoader().load_and_split(json_file)

    logger.info("creating vectorstore...")
    vectorstore = FAISS.from_documents(docs, embedding=OpenAIEmbeddings())

    return vectorstore
