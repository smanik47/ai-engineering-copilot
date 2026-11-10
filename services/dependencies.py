from services.document_service import DocumentService
from services.chat_service import ChatService


# Create shared service instances

_document_service = DocumentService()
_chat_service = ChatService()


# Dependency providers

def get_document_service():

    return _document_service


def get_chat_service():

    return _chat_service