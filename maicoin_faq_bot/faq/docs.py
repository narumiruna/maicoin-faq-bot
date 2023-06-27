from typing import List

from langchain.schema import Document
from loguru import logger

from maicoin_faq_bot.utils import load_json


class FAQDocuments(List[Document]):

    @staticmethod
    def from_json(json_file: str) -> List[Document]:
        logger.info(f'loading json file: {json_file}')
        data = load_json(json_file)
        logger.info('creating documents...')

        docs = []
        for d in data:
            page_content = (f'Title: {d["title"]}\n'
                            f'URL: {d["url"]}\n'
                            f'{d["body"]}')
            docs.append(Document(page_content=page_content))

        return docs
