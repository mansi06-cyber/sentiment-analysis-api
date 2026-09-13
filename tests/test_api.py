"""Unit and integration tests for the Sentiment Analysis API."""
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.schemas import count_words


@pytest.fixture(scope="session")
def client():
    """Create a TestClient with lifespan events executed."""
    with TestClient(app) as test_client:
        yield test_client


def test_count_words_helper():
    """Test the word counter utility."""
    assert count_words("hello world") == 2
    assert count_words("   spaces   around   words   ") == 3
    assert count_words("") == 0


def test_health_check(client):
    """Test the GET /api/health endpoint."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "model_name" in data
    assert "device" in data
    assert data["status"] in ("healthy", "loading")


def test_analyze_positive_sentiment(client):
    """Test POST /api/analyze with positive text."""
    payload = {"text": "I absolutely love this new product! It works wonderfully and makes me so happy."}
    response = client.post("/api/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["text"] == payload["text"]
    assert data["sentiment"] == "positive"
    assert 0.0 <= data["confidence"] <= 1.0
    assert data["word_count"] > 0


def test_analyze_negative_sentiment(client):
    """Test POST /api/analyze with negative text."""
    payload = {"text": "This was the worst experience of my life. Completely broken and frustrating."}
    response = client.post("/api/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["text"] == payload["text"]
    assert data["sentiment"] == "negative"
    assert 0.0 <= data["confidence"] <= 1.0
    assert data["word_count"] > 0


def test_analyze_neutral_sentiment(client):
    """Test POST /api/analyze with neutral text."""
    payload = {"text": "The meeting is scheduled for 3 PM on Thursday in conference room B."}
    response = client.post("/api/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["text"] == payload["text"]
    assert data["sentiment"] == "neutral"
    assert 0.0 <= data["confidence"] <= 1.0
    assert data["word_count"] > 0


def test_analyze_validation_empty_text(client):
    """Test POST /api/analyze with empty text should return 422."""
    response = client.post("/api/analyze", json={"text": "   "})
    assert response.status_code == 422
    data = response.json()
    assert "error" in data or "detail" in data


def test_analyze_validation_exceeds_500_words(client):
    """Test POST /api/analyze with more than 500 words should return 422."""
    long_text = "word " * 501
    response = client.post("/api/analyze", json={"text": long_text})
    assert response.status_code == 422
    data = response.json()
    assert "error" in data or "detail" in data


def test_analyze_validation_missing_field(client):
    """Test POST /api/analyze without required 'text' field should return 422."""
    response = client.post("/api/analyze", json={})
    assert response.status_code == 422


def test_batch_analyze_success(client):
    """Test POST /api/analyze/batch with valid texts."""
    payload = {
        "texts": [
            "I am thrilled with these fantastic results!",
            "The meeting is scheduled for tomorrow at noon.",
            "This service is awful and terrible."
        ]
    }
    response = client.post("/api/analyze/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 3
    assert len(data["results"]) == 3

    assert data["results"][0]["sentiment"] == "positive"
    assert data["results"][1]["sentiment"] == "neutral"
    assert data["results"][2]["sentiment"] == "negative"

    for item in data["results"]:
        assert 0.0 <= item["confidence"] <= 1.0
        assert item["word_count"] > 0


def test_batch_analyze_empty_list(client):
    """Test POST /api/analyze/batch with empty list should return 422."""
    response = client.post("/api/analyze/batch", json={"texts": []})
    assert response.status_code == 422


def test_batch_analyze_exceeds_max_10_texts(client):
    """Test POST /api/analyze/batch with 11 texts should return 422."""
    texts = [f"Text sample number {i}" for i in range(11)]
    response = client.post("/api/analyze/batch", json={"texts": texts})
    assert response.status_code == 422


def test_batch_analyze_item_too_long(client):
    """Test POST /api/analyze/batch where one item exceeds 500 words should return 422."""
    long_item = "word " * 501
    payload = {
        "texts": [
            "A short valid text.",
            long_item
        ]
    }
    response = client.post("/api/analyze/batch", json=payload)
    assert response.status_code == 422
