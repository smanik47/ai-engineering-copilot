from unittest.mock import patch

from services.chat_service import ChatService


def test_chat_service_returns_answer():

    expected_result = {
        "answer": "Redis is an in-memory data store.",
        "sources": []
    }

    with patch(
            "services.chat_service.answer_question",
            return_value=expected_result
    ) as mock_answer:

        service = ChatService()

        result = service.ask_question(
            "What is Redis?",
            3
        )

        assert result == expected_result

        mock_answer.assert_called_once_with(
            "What is Redis?",
            3
        )