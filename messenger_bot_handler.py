from flask import Flask, request
import requests
import os
from dotenv import load_dotenv

from chat_memory_service import ChatMemoryService
from chatbot import chat_with_rag_future
import httpx

load_dotenv()

app = Flask(__name__)

# These will come from your Facebook Developer Console
ACCESS_TOKEN = os.getenv('META_PAGE_ACCESS_TOKEN')
VERIFY_TOKEN = os.getenv('META_VERIFY_TOKEN') # You invent this string

chat_memory_service = ChatMemoryService()

@app.route('/', methods=['GET'])
def verify():
    # Webhook verification (Facebook sends a GET request to verify your server)
    if request.args.get("hub.mode") == "subscribe" and request.args.get("hub.challenge"):
        if not request.args.get("hub.verify_token") == VERIFY_TOKEN:
            return "Verification token mismatch", 403
        return request.args["hub.challenge"], 200
    return "Hello world", 200

@app.route('/', methods=['POST'])
async def webhook():
    # Handle incoming messages
    data = request.get_json()

    if data["object"] == "page":
        for entry in data["entry"]:
            for messaging_event in entry["messaging"]:
                if messaging_event.get("message"):
                    sender_id = messaging_event["sender"]["id"]
                    message_text = messaging_event["message"]["text"]

                    # show typing here
                    url = f"https://graph.facebook.com/v18.0/me/messages?access_token={ACCESS_TOKEN}"
                    typing_payload = {
                        "recipient": {"id": sender_id},
                        "sender_action": "typing_on"
                    }
                    async with httpx.AsyncClient() as client:
                        await client.post(url, json=typing_payload)

                    # await this
                    response = await chat_with_rag_future(
                        prompt=message_text,
                        previous_message=chat_memory_service.get_memory_for_user(sender_id).to_string()
                    )


                    # Echo the message back
                    await send_message(sender_id, response)

                    chat_memory_service.add_user_message(sender_id, message_text)
                    chat_memory_service.add_bot_message(sender_id, response)

    return "ok", 200

async def send_message(recipient_id, text):
    url = f"https://graph.facebook.com/v18.0/me/messages?access_token={ACCESS_TOKEN}"
    payload = {
        "recipient": {"id": recipient_id},
        "message": {"text": text}
    }
    async with httpx.AsyncClient() as client:
        await client.post(url, json=payload)

if __name__ == "__main__":
    app.run(port=5000, debug=True)