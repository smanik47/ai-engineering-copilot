from fastapi import FastAPI

from main import generate_ai_response

from pydantic import BaseModel


class ChatRequest(BaseModel):
    question: str

app = FastAPI()


@app.get("/")
def home():
    return {"message": "Welcome to AI Engineering Copilot!"}


@app.get("/health", include_in_schema=True)
def health():
    return {"status": "UP"}


# Debug registered routes
for route in app.routes:
    print(
        f"Path: {route.path}, "
        f"Methods: {getattr(route, 'methods', None)}, "
        f"Included in schema: {getattr(route, 'include_in_schema', None)}"
    )

@app.post("/chat")
def chat(request: ChatRequest):

    answer = generate_ai_response(request.question)

    return {
        "question": request.question,
        "answer": answer
    }