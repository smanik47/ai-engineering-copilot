
import redis
from dotenv import load_dotenv
from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")

REDIS_URL = os.getenv(
    "REDIS_URL",
    "redis://localhost:6379/0"
)


def get_redis_client():

    return redis.Redis.from_url(
        REDIS_URL,
        decode_responses=True
    )
