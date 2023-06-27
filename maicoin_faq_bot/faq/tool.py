from langchain.base_language import BaseLanguageModel
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.tools import Tool
from langchain.vectorstores.base import VectorStoreRetriever


def load_faq_tool(llm: BaseLanguageModel, retriever: VectorStoreRetriever, max_output_chars: int = 4000):
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=ConversationBufferWindowMemory(memory_key='chat_history', return_messages=True),
        verbose=True,
    )

    def _run(query):
        res = chain.run(query)[:max_output_chars]
        print(res)
        return res

    return Tool.from_function(
        name='MaiCoin-FAQ-chain',
        description=('Useful for when you need to answer questions. '
                     'You should ALWAYS use this. '
                     'Input should be a fully formed question.'),
        func=_run,
    )
