from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import os
from dotenv import load_dotenv
load_dotenv()

# Define the database URL for the AWS RDS MySQL database
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
SQLALCHEMY_DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Create an engine that connects to the RDS MySQL database
engine = create_engine(SQLALCHEMY_DATABASE_URL)

# Create a configured "Session" class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create a Session object that can be used to interact with the database
session = Session(engine)

# Base class for declarative class definitions
Base = declarative_base()

# Dependency to get a database session
def get_db():
    db = SessionLocal()  # Instantiate a new database session
    try:
        yield db  # Yield the session to be used in the calling function
    finally:
        db.close()  # Ensure that the session is closed after use

# Function to check if the database is connected
def is_db_connected():
    try:
        # Attempt to connect to the database
        with engine.connect() as connection:
            return "Database connection successful"
    except Exception as e:
        return f"Database connection failed: {e}"

# Example usage
if __name__ == "__main__":
    print(is_db_connected())