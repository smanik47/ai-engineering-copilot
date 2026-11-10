import pytest

from fastapi.testclient import TestClient

from app import app
from services.dependencies import get_chat_service


class FakeChatService:

    def ask_question(self, question, top_k):

        return {
            "answer": "Redis is an in-memory data store.",
            "sources": [
                {
                    "chunk_id": "redis_p1_c0",
                    "source": "database.pdf",
                    "page": 1,
                    "similarity_score": 0.91
                }
            ]
        }


@pytest.fixture
def client():

    app.dependency_overrides[get_chat_service] = (
        lambda: FakeChatService()
    )

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


def test_chat_endpoint_success(client):

    response = client.post(
        "/chat",
        json={
            "question": "What is Redis?",
            "top_k": 3
        }
    )
    print("STATUS:", response.status_code)
    print("RESPONSE:", response.text)

    assert response.status_code == 200

    data = response.json()

    assert data["question"] == "What is Redis?"

    assert data["answer"] == (
        "Redis is an in-memory data store."
    )

    assert data["sources"][0]["source"] == "database.pdf"


def test_chat_endpoint_invalid_request(client):

    response = client.post(
        "/chat",
        json={
            "question": "",
            "top_k": 3
        }
    )

    assert response.status_code == 422