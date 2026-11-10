
import json

from services.redis_client import get_redis_client


class MemoryService:

    def __init__(self):

        self.redis_client = get_redis_client()

    def get_conversation_history(self, conversation_id: str):

        key = f"conversation:{conversation_id}"

        messages = self.redis_client.lrange(
            key,
            0,
            -1
        )

        return [
            json.loads(message)
            for message in messages
        ]

    def save_message(self, conversation_id: str, question: str, answer: str):

        key = f"conversation:{conversation_id}"

        messages = [
            {
                "role": "user",
                "content": question
            },
            {
                "role": "assistant",
                "content": answer
            }
        ]

        pipeline = self.redis_client.pipeline()

        for message in messages:

            pipeline.rpush(
                key,
                json.dumps(message)
            )

        # Keep the latest 20 messages
        pipeline.ltrim(
            key,
            -20,
            -1
        )

        # Expire conversation after 7 days
        pipeline.expire(
            key,
            7 * 24 * 60 * 60
        )

        pipeline.execute()
