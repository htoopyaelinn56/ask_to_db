# to hold two previous chat messages in memory (one for user and one for bot)
class ChatMemory:
    def __init__(self):
        self.user_message = None
        self.bot_message = None

    def add_user_message(self, message):
        self.user_message = message

    def add_bot_message(self, message):
        self.bot_message = message

    def get_memory(self):
        memory = []
        if self.user_message:
            memory.append({"role": "user", "content": self.user_message})
        if self.bot_message:
            memory.append({"role": "bot", "content": self.bot_message})
        return memory

    def to_string(self) -> str:
        if not self.user_message and not self.bot_message:
            return ""
        parts = ["Chat Memory (2 previous message):"]
        if self.user_message:
            parts.append(f"User: {self.user_message}")
        if self.bot_message:
            parts.append(f"Bot: {self.bot_message}")
        return "\n".join(parts)


# user_id to ChatMemory
class ChatMemoryService:
    def __init__(self):
        self.memory_store: dict[str, ChatMemory] = {}

    def get_memory_for_user(self, user_id) -> ChatMemory:
        if user_id not in self.memory_store:
            self.memory_store[user_id] = ChatMemory()
        return self.memory_store[user_id]

    def add_user_message(self, user_id, message):
        chat_memory = self.get_memory_for_user(user_id)
        chat_memory.add_user_message(message)

    def add_bot_message(self, user_id, message):
        chat_memory = self.get_memory_for_user(user_id)
        chat_memory.add_bot_message(message)

