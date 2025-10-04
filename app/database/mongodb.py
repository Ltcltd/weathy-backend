#handles database connection

import motor.motor_asyncio

MONGO_URI = "mongodb://localhost:27017"


def main():
    client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)  #connnect to local db instance

    db = client["weathy"]
    collection = db["trends"]

main()