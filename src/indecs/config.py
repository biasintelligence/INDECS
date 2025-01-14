"""Config to keep track of all system specific variables."""
from os import environ, path
from dotenv import load_dotenv

BASE_DIR = path.abspath(path.join(path.dirname(__file__), "."))
load_dotenv(path.join(BASE_DIR, ".env"))

class Config:
    """Configuration variables"""

    # Data locations
    ANNOTATIONS_PATH = environ.get("ANNOTATIONS_PATH")
    MODELS_PATH=environ.get("MODELS_PATH")
    VIDEO_PATH = environ.get("VIDEO_PATH")


