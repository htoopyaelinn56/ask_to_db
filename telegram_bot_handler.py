from dotenv import load_dotenv
from telegram.constants import ParseMode
import telegramify_markdown
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters
import os

from chat_memory_service import ChatMemoryService
from chatbot import chat_with_rag_stream

load_dotenv()
import asyncio
from telegram import Update
from telegram.ext import ContextTypes

chat_memory_service = ChatMemoryService()


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text

    # 1. Send an initial placeholder message
    placeholder = await update.message.reply_text("Thinking...")

    full_response = ""
    chunk_counter = 0

    # 2. Iterate through your RAG generator
    # Assuming chat_with_rag_stream is defined as discussed previously
    try:
        for chunk in chat_with_rag_stream(prompt=user_text, previous_message=chat_memory_service.get_memory_for_user(
                update.effective_chat.id).to_string()):
            full_response += chunk
            chunk_counter += 1

            # 3. Update the message every 15 chunks to avoid rate limits
            if chunk_counter % 15 == 0:
                try:
                    await context.bot.edit_message_text(
                        chat_id=update.effective_chat.id,
                        message_id=placeholder.message_id,
                        text=full_response + " "  # Visual cursor
                    )
                    # Small sleep to respect Telegram's rate limits
                    await asyncio.sleep(0.1)
                except Exception:
                    pass  # Ignore errors like "Message is not modified"
    # catch errors from the RAG generator
    except Exception as e:
        await context.bot.edit_message_text(
            chat_id=update.effective_chat.id,
            message_id=placeholder.message_id,
            text="An error occurred while processing your request. Please try again later."
        )
        return

    # 4. Final update to remove the cursor and show complete text
    print("[DEBUG] Full response to Telegram Bot:", full_response)
    chat_memory_service.add_user_message(update.effective_chat.id, user_text)
    chat_memory_service.add_bot_message(update.effective_chat.id, full_response)
    await context.bot.edit_message_text(
        chat_id=update.effective_chat.id,
        message_id=placeholder.message_id,
        text=telegramify_markdown.markdownify(full_response),
        parse_mode=ParseMode.MARKDOWN_V2
    )


if __name__ == '__main__':
    # Replace 'YOUR_TOKEN_HERE' with the token from BotFather
    app = ApplicationBuilder().token(os.getenv("TELEGRAM_BOT_TOKEN")).build()

    # Add a handler that filters for all text messages
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))

    print("Bot is running...")
    app.run_polling()
