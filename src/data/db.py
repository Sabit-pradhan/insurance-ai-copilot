# src/data/db.py

import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL


# Load latest values from .env
load_dotenv(override=True)


# Read database configuration
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")


# Create PostgreSQL URL safely
# This handles special characters in passwords automatically
DATABASE_URL = URL.create(
    drivername="postgresql+psycopg2",
    username=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=int(DB_PORT),
    database=DB_NAME
)


# Create connection engine
engine = create_engine(DATABASE_URL)


def test_connection():

    # Connect to PostgreSQL
    with engine.connect() as connection:

        # Simple query to verify connection
        result = connection.execute(
            text("SELECT 1")
        )

        return result.scalar()