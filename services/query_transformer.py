
from openai import OpenAI


class QueryTransformer:

    def __init__(self):

        self.client = OpenAI()

    def rewrite_query(self, question: str, chat_history=None):

        history = chat_history or []

        prompt = f"""
        You are a query rewriting assistant.

        Your task is to rewrite the user's question
        into a clear and search-friendly query for a
        vector database.

        Rules:
        1. Preserve the original intent.
        2. Use relevant technical terminology.
        3. Use conversation history when necessary.
        4. Do not answer the question.
        5. Do not introduce unsupported assumptions.

        Conversation history:
        {history}

        Original question:
        {question}

        Return only the rewritten query.
        """

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0
        )

        rewritten_query = (
            response.choices[0].message.content
        )

        return rewritten_query.strip()


    def generate_multiple_queries(self, question: str):

        prompt = f"""
        You are a search query generation assistant.

        Generate 3 different search queries that explore
        different aspects of the user's original question.

        Rules:
        1. Preserve the user's intent.
        2. Each query should explore a different angle.
        3. Use technical terminology where appropriate.
        4. Do not answer the question.
        5. Return exactly 3 queries, one per line.
        6. Do not include numbering or explanations.

        Original question:
        {question}
        """

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0
        )

        content = response.choices[0].message.content

        queries = [
            line.strip()
            for line in content.splitlines()
            if line.strip()
        ]

        return queries[:3]
