"""Pydantic schemas for request and response validation."""
from enum import Enum
from pydantic import BaseModel, Field, field_validator
from app.config import MIN_WORDS, MAX_WORDS, MAX_BATCH_SIZE


def count_words(text: str) -> int:
    """Helper to count whitespace-separated words in a string."""
    return len(text.strip().split())


class SentimentType(str, Enum):
    """Supported sentiment classifications."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class AnalyzeRequest(BaseModel):
    """Request schema for single-text sentiment analysis."""
    text: str = Field(
        ...,
        description=f"English text to analyze ({MIN_WORDS} to {MAX_WORDS} words)",
        examples=["I am really excited about this internship project!"]
    )

    @field_validator("text")
    @classmethod
    def validate_word_count(cls, v: str) -> str:
        if not isinstance(v, str):
            raise ValueError("Text must be a string.")
        cleaned = v.strip()
        words = count_words(cleaned)
        if words < MIN_WORDS:
            raise ValueError(
                f"Text is too short. It must contain at least {MIN_WORDS} word."
            )
        if words > MAX_WORDS:
            raise ValueError(
                f"Text is too long. It must not exceed {MAX_WORDS} words. (Received: {words} words)"
            )
        return cleaned


class AnalyzeResponse(BaseModel):
    """Response schema for single-text sentiment analysis."""
    text: str = Field(..., description="Input text analyzed")
    sentiment: SentimentType = Field(
        ...,
        description="Sentiment classification: positive, negative, or neutral"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0"
    )
    word_count: int = Field(..., description="Word count of the input text")


class BatchAnalyzeRequest(BaseModel):
    """Request schema for batch sentiment analysis."""
    texts: list[str] = Field(
        ...,
        min_length=1,
        max_length=MAX_BATCH_SIZE,
        description=f"List of English texts (1 to {MAX_BATCH_SIZE} items, {MIN_WORDS}-{MAX_WORDS} words each)",
        examples=[
            [
                "The product exceeded all my expectations!",
                "The flight was delayed by 20 minutes.",
                "Worst customer support experience ever."
            ]
        ]
    )

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("Batch cannot be empty.")
        if len(v) > MAX_BATCH_SIZE:
            raise ValueError(
                f"Batch size exceeds maximum allowed limit of {MAX_BATCH_SIZE} texts."
            )

        validated = []
        for idx, item in enumerate(v):
            if not isinstance(item, str):
                raise ValueError(f"Item at index {idx} must be a string.")
            cleaned = item.strip()
            words = count_words(cleaned)
            if words < MIN_WORDS:
                raise ValueError(
                    f"Item at index {idx} is empty or has 0 words. Must have at least {MIN_WORDS} word."
                )
            if words > MAX_WORDS:
                raise ValueError(
                    f"Item at index {idx} has {words} words, exceeding limit of {MAX_WORDS} words."
                )
            validated.append(cleaned)
        return validated


class BatchAnalyzeResponse(BaseModel):
    """Response schema for batch sentiment analysis."""
    total: int = Field(..., description="Total number of items analyzed")
    results: list[AnalyzeResponse] = Field(..., description="Analysis result for each text")


class HealthResponse(BaseModel):
    """Response schema for system and model health status."""
    status: str = Field(..., description="Service status", examples=["healthy"])
    model_loaded: bool = Field(..., description="Indicates if sentiment model is loaded in memory")
    model_name: str = Field(..., description="Hugging Face model identifier")
    device: str = Field(..., description="Computation device in use (cpu or cuda)")
