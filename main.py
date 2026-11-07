import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env
load_dotenv()

# Read the API key
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY is not configured")

# Create OpenAI client
client = OpenAI(api_key=api_key)

def generate_ai_response(question: str):

    response = client.responses.create(
        model="gpt-5.5",
        input=question
    )

    return response.output_text

