from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools import BaseTool
from langchain.vectorstores import Chroma
from langchain.vectorstores.base import VectorStoreRetriever

from .utils import load_json


class MaiCoinFAQRetriever(BaseTool):
    name: str = 'get_relevant_documents'
    description: str = ('A MaiCoin FAQ article retriever.'
                        'Input a query string.'
                        'Output MaiCoin FAQ article(s) that are relevant to the query string.')

    retriever: VectorStoreRetriever
    max_output_chars: int = 4000

    def _run(self, query: str) -> str:
        docs = self.retriever.get_relevant_documents(query)

        outputs = []
        for doc in docs:
            outputs.append((f'Query: {query}\n'
                            f'Page Title: {doc.metadata["title"]}\n'
                            f'Page URL: {doc.metadata["url"]}\n'
                            f'Page Content: {doc.page_content}\n'))

        return '\n\n'.join(outputs)[:self.max_output_chars]

    async def _arun(self, query: str) -> str:
        return self._run(query)

    @classmethod
    def from_json(cls, f: str):
        data = load_json(f)
        docs = RecursiveCharacterTextSplitter().split_documents(
            [Document(page_content=d['body'], metadata={
                'title': d['title'],
                'url': d['url']
            }) for d in data])
        vectorstore = Chroma.from_documents(docs, embedding=OpenAIEmbeddings())
        return cls(retriever=vectorstore.as_retriever())
