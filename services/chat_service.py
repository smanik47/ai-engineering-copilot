from query import answer_question


class ChatService:

    def ask_question(self, question: str):

        result = answer_question(question)

        return result