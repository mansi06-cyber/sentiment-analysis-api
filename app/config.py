"""Application configuration settings."""
import os

# Model configuration
# Cardiff NLP twitter-roberta-base-sentiment-latest is a 3-class sentiment model:
# returns labels: 'negative', 'neutral', 'positive'
MODEL_NAME: str = os.getenv("MODEL_NAME", "cardiffnlp/twitter-roberta-base-sentiment-latest")

# Input validation constraints
MIN_WORDS: int = 1
MAX_WORDS: int = 500
MAX_BATCH_SIZE: int = 10

# API Metadata
API_TITLE: str = "Sentiment Analysis API"
API_DESCRIPTION: str = (
    "Production-ready Sentiment Analysis REST API using FastAPI, "
    "Hugging Face Transformers, and PyTorch."
)
API_VERSION: str = "1.0.0"
