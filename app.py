from pathlib import Path
from uuid import uuid4

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException,
    Depends
)

from schemas import ChatRequest
from services.document_service import DocumentService
from services.chat_service import ChatService

from dependencies import (
    get_document_service,
    get_chat_service
)


# --------------------------------
# APPLICATION INITIALIZATION
# --------------------------------

app = FastAPI(
    title="AI Engineering Knowledge Copilot",
    description="A RAG-based technical knowledge assistant",
    version="1.0.0"
)

BASE_DIR = Path(__file__).resolve().parent

PDF_DIR = BASE_DIR / "knowledge" / "pdfs"

PDF_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------
# HEALTH ENDPOINT
# --------------------------------

@app.get("/health")
def health():

    return {
        "status": "UP"
    }


# --------------------------------
# DOCUMENT UPLOAD ENDPOINT
# --------------------------------

@app.post("/documents/upload")
def upload_document(
        file: UploadFile = File(...),
        service: DocumentService = Depends(get_document_service)
):

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is required"
        )

    if Path(file.filename).suffix.lower() != ".pdf":
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )

    contents = file.file.read()

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty"
        )

    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="File size cannot exceed 10 MB"
        )

    stored_filename = f"{uuid4().hex}.pdf"

    file_path = PDF_DIR / stored_filename

    file_path.write_bytes(contents)

    try:

        result = service.ingest_documents()

        return {
            "message": "Document processed successfully",
            "filename": file.filename,
            "stored_filename": stored_filename,
            "total_chunks": result["total_chunks"]
        }

    except Exception:

        file_path.unlink(missing_ok=True)

        raise HTTPException(
            status_code=500,
            detail="Document ingestion failed"
        )


# --------------------------------
# CHAT ENDPOINT
# --------------------------------

@app.post("/chat")
def chat(
        request: ChatRequest,
        service: ChatService = Depends(get_chat_service)
):

    try:

        result = service.ask_question(
            request.question
        )

        return {
            "question": request.question,
            "answer": result["answer"],
            "sources": result["sources"]
        }

    except FileNotFoundError:

        raise HTTPException(
            status_code=404,
            detail="Knowledge base not found. Upload a PDF first."
        )

    except Exception:

        raise HTTPException(
            status_code=500,
            detail="Unable to generate answer"
        )