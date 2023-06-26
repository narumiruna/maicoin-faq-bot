import html
import json
import os
import traceback

from dotenv import load_dotenv
from loguru import logger
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import ApplicationBuilder
from telegram.ext import ContextTypes
from telegram.ext import MessageHandler
from telegram.ext import filters

from .agent import MaiCoinFAQAgent


def start_bot():
    load_dotenv('.env')

    bot_token = os.environ.get('BOT_TOKEN')
    if bot_token is None:
        raise ValueError('BOT_TOKEN is not set')

    app = ApplicationBuilder().token(bot_token).build()

    developer_chat_id = os.environ.get('DEVELOPER_CHAT_ID')
    logger.info('developer_chat_id: {}', developer_chat_id)

    async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Log the error and send a telegram message to notify the developer."""
        # Log the error before we do anything else, so we can see it even if something breaks.
        logger.error("Exception while handling an update:", exc_info=context.error)

        # traceback.format_exception returns the usual python message about an exception, but as a
        # list of strings rather than a single string, so we have to join them together.
        tb_list = traceback.format_exception(None, context.error, context.error.__traceback__)
        tb_string = "".join(tb_list)

        # Build the message with some markup and additional information about what happened.
        # You might need to add some logic to deal with messages longer than the 4096 character limit.
        update_str = update.to_dict() if isinstance(update, Update) else str(update)
        message = (f"An exception was raised while handling an update\n"
                   f"<pre>update = {html.escape(json.dumps(update_str, indent=2, ensure_ascii=False))}"
                   "</pre>\n\n"
                   f"<pre>context.chat_data = {html.escape(str(context.chat_data))}</pre>\n\n"
                   f"<pre>context.user_data = {html.escape(str(context.user_data))}</pre>\n\n"
                   f"<pre>{html.escape(tb_string)}</pre>")

        # Finally, send the message
        await context.bot.send_message(chat_id=developer_chat_id if developer_chat_id else update.effective_chat.id,
                                       text=message,
                                       parse_mode=ParseMode.HTML)

        await context.bot.send_message(chat_id=update.effective_chat.id,
                                       text='Something went wrong. Please try again later.')

    app.add_error_handler(error_handler)

    # add langchain bot
    agent = MaiCoinFAQAgent()
    app.add_handler(MessageHandler(filters=filters.TEXT, callback=agent.chat))

    app.run_polling()
