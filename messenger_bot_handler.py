from flask import Flask, request
import requests
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# These will come from your Facebook Developer Console
ACCESS_TOKEN = os.getenv('META_PAGE_ACCESS_TOKEN')
VERIFY_TOKEN = os.getenv('META_VERIFY_TOKEN') # You invent this string

@app.route('/', methods=['GET'])
def verify():
    # Webhook verification (Facebook sends a GET request to verify your server)
    if request.args.get("hub.mode") == "subscribe" and request.args.get("hub.challenge"):
        if not request.args.get("hub.verify_token") == VERIFY_TOKEN:
            return "Verification token mismatch", 403
        return request.args["hub.challenge"], 200
    return "Hello world", 200

@app.route('/', methods=['POST'])
def webhook():
    # Handle incoming messages
    data = request.get_json()

    if data["object"] == "page":
        for entry in data["entry"]:
            for messaging_event in entry["messaging"]:
                if messaging_event.get("message"):
                    sender_id = messaging_event["sender"]["id"]
                    message_text = messaging_event["message"]["text"]

                    # Echo the message back
                    send_message(sender_id, f"You said: {message_text}")

    return "ok", 200

def send_message(recipient_id, text):
    url = f"https://graph.facebook.com/v18.0/me/messages?access_token={ACCESS_TOKEN}"
    payload = {
        "recipient": {"id": recipient_id},
        "message": {"text": text}
    }
    requests.post(url, json=payload)

if __name__ == "__main__":
    app.run(port=5000, debug=True)