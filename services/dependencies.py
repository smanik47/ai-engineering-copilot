
from services.document_service import DocumentService
from services.chat_service import ChatService
from services.memory_service import MemoryService


# Shared service instances

_document_service = DocumentService()

_memory_service = MemoryService()

_chat_service = ChatService(
    memory_service=_memory_service
)


# Dependency providers

def get_document_service():

    return _document_service


def get_chat_service():

    return _chat_service
