from __future__ import annotations

import os

from dotenv import load_dotenv
from loguru import logger
from telegram.ext import ApplicationBuilder
from telegram.ext import MessageHandler
from telegram.ext import filters

from .agent import MaiCoinFAQAgent
from .error import ErrorHandler


def start_bot():
    load_dotenv('.env')

    bot_token = os.environ.get('BOT_TOKEN')
    if bot_token is None:
        raise ValueError('BOT_TOKEN is not set')

    app = ApplicationBuilder().token(bot_token).build()

    developer_chat_id = os.environ.get('DEVELOPER_CHAT_ID')
    logger.info('developer_chat_id: {}', developer_chat_id)

    app.add_error_handler(ErrorHandler.from_env())

    # add langchain bot
    agent = MaiCoinFAQAgent.from_env()
    app.add_handler(MessageHandler(filters=filters.TEXT & (~filters.COMMAND), callback=agent.chat))

    app.run_polling()
