from query import answer_question


class ChatService:

    def __init__(self, memory_service):

        self.memory_service = memory_service

    def ask_question(
            self,
            question: str,
            conversation_id: str,
            top_k: int = 3
    ):

        # 1. Retrieve previous conversation
        history = self.memory_service.get_conversation_history(
            conversation_id
        )

        # 2. Generate answer using RAG + conversation history
        result = answer_question(
            question,
            top_k,
            history
        )

        # 3. Save current question and generated answer
        self.memory_service.save_message(
            conversation_id,
            question,
            result["answer"]
        )

        return result
