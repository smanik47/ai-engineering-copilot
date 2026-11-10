import redis

# Connect to Redis server
r = redis.Redis(
    host="localhost",
    port=6379,
    decode_responses=True
)

# Check connection
print(r.ping())

# Store data
r.set("username", "Manik")

# Retrieve data
value = r.get("username")

print(value)