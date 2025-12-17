# Ask to DB

Simple demo app that lets you chat with a product database using AI and embeddings.

## Prerequisites

- Python 3 installed
- Required Python packages installed (check imports in `ai_service.py`, `db_service.py`, `embedding_service.py`, `chatbot.py` and install via pip as needed)
- SQLite or the database configured in `db_service.py`

## 1. Set up the database

1. Create the database defined in `schema.sql`.
2. Load initial data:

```bash
sqlite3 your_database.db < schema.sql
sqlite3 your_database.db < seed.sql
```

> Adjust the commands if you are using a different database engine; follow the connection details in `db_service.py`.

## 2. Generate embeddings and feed data

Run the DB/embedding service to read products and store their embeddings (as implemented in your code):

```bash
python db_service.py
```

This step should populate any embedding-related tables/columns used by the chatbot.

## 3. Run the chatbot app

Start the main chatbot:

```bash
python chatbot.py
```

Interact with the app as instructed in the console.

## Optional: Telegram bot

If configured, you can run the Telegram bot entrypoint:

```bash
python telegram_bot_handler.py
```

Make sure you have set the required environment variables (e.g. bot token) before running.
