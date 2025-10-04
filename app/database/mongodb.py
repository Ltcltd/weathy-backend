#handles database connection

import os
import motor.motor_asyncio 

MONGO_URI = os.getenv("DATABASE_URL", "mongodb://localhost:27017")

client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
db = client["weathyDB"] 

def get_database():
    return db 