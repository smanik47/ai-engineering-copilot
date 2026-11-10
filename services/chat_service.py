from query import answer_question


class ChatService:

    def ask_question(self, question: str, top_k: int = 3):

        result = answer_question(
            question,
            top_k
        )

        return result