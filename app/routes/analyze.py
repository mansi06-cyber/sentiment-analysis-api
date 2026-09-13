"""Sentiment analysis endpoints router."""
from fastapi import APIRouter, HTTPException, status
from app.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    BatchAnalyzeRequest,
    BatchAnalyzeResponse,
    SentimentType,
    count_words,
)
from app.services import sentiment_service

router = APIRouter(tags=["Sentiment Analysis"])


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyze Single Text",
    description="Classifies the sentiment of a single English text (1 to 500 words) as positive, negative, or neutral."
)
async def analyze_text(payload: AnalyzeRequest) -> AnalyzeResponse:
    """Analyze sentiment of a single English text."""
    if not sentiment_service.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is currently initializing. Please try again shortly."
        )

    try:
        sentiment, confidence = sentiment_service.predict(payload.text)
        return AnalyzeResponse(
            text=payload.text,
            sentiment=SentimentType(sentiment),
            confidence=confidence,
            word_count=count_words(payload.text)
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(exc)}"
        ) from exc


@router.post(
    "/analyze/batch",
    response_model=BatchAnalyzeResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyze Batch of Texts",
    description="Classifies a batch of up to 10 English texts (1 to 500 words each)."
)
async def analyze_batch(payload: BatchAnalyzeRequest) -> BatchAnalyzeResponse:
    """Analyze sentiment for a list of English texts (maximum 10 texts)."""
    if not sentiment_service.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is currently initializing. Please try again shortly."
        )

    try:
        predictions = sentiment_service.predict_batch(payload.texts)
        results: list[AnalyzeResponse] = []
        for text, (sentiment, confidence) in zip(payload.texts, predictions):
            results.append(
                AnalyzeResponse(
                    text=text,
                    sentiment=SentimentType(sentiment),
                    confidence=confidence,
                    word_count=count_words(text)
                )
            )

        return BatchAnalyzeResponse(
            total=len(results),
            results=results
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch inference error: {str(exc)}"
        ) from exc
