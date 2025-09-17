import json
from channels.generic.websocket import WebsocketConsumer
from .services.chat_services_streaming import chatbot_graph
from langchain_core.messages import HumanMessage

class ChatConsumer(WebsocketConsumer):
    def connect(self):
        print("WebSocket connected")
        self.accept()
        self.send(text_data=json.dumps({
            "sender": "bot",
            "avatar": "🤖",
            "message": "Hello — I am a demo chatbot UI. Try typing a message below.",
            "type": "init"
        }))

    def disconnect(self, close_code):
        pass

    def receive(self, text_data):
        data = json.loads(text_data)
        user_message = data.get("message", "")
        user_sender = data.get("sender")

        # echo user message
        self.send(text_data=json.dumps({
            "sender": user_sender,
            "message": user_message,
            "type": "user"
        }))

        # stream bot response
        for message_chunk, _ in chatbot_graph.stream(
            {"messages": [HumanMessage(content=user_message)]},
            config={"configurable": {"thread_id": "1"}},
            stream_mode="messages",
        ):
            if message_chunk.content:
                self.send(text_data=json.dumps({
                    "sender": "bot",
                    "avatar": "🤖",
                    "message": message_chunk.content,
                    "type": "stream"
                }))

        # 🔑 just mark end, don’t resend text
        self.send(text_data=json.dumps({
            "sender": "bot",
            "avatar": "🤖",
            "type": "done"
        }))
