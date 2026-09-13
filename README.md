# Sentiment Analysis API

A clean, beginner-friendly REST API for sentiment analysis built with **FastAPI**, **Hugging Face Transformers**, and **PyTorch**.

The API uses the pre-trained [`cardiffnlp/twitter-roberta-base-sentiment-latest`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest) model, which accurately classifies English text into **three distinct sentiments**:
- `positive`
- `negative`
- `neutral`

Along with the classified sentiment, each prediction includes a **confidence score** between `0.0` and `1.0`.

---

## Features

- **Single Text Analysis (`POST /api/analyze`)**:
  - Classifies sentiment of an English text.
  - Validates text length to ensure it contains between **1 and 500 words**.
  - Returns sentiment, confidence score, and word count.

- **Batch Analysis (`POST /api/analyze/batch`)**:
  - Analyzes multiple texts in a single request (maximum **10 texts**).
  - Performs batched tensor inference for faster execution.
  - Enforces word count constraints for each individual item.

- **Health Check (`GET /api/health`)**:
  - Provides real-time status of the API, model load state, model identifier, and hardware device (`cpu` or `cuda`).

- **Input Validation & Error Handling**:
  - Strict Pydantic validators returning clear, structured HTTP `422` error messages when inputs are out of bounds (e.g. 0 words, >500 words, or >10 items in a batch).

---

## Project Structure

```
sentiment-analysis-api/
├── app/
│   ├── __init__.py           # Package marker
│   ├── config.py             # App configurations and constants
│   ├── main.py               # FastAPI app, lifespan, CORS, and error handlers
│   ├── schemas.py            # Pydantic request/response models & validation
│   ├── services.py           # Model loading & PyTorch inference engine
│   └── routes/
│       ├── __init__.py
│       ├── analyze.py        # /api/analyze & /api/analyze/batch endpoints
│       └── health.py         # /api/health endpoint
├── tests/
│   ├── __init__.py
│   └── test_api.py           # Pytest unit and integration tests
├── .gitignore                # Ignored files (venv, pycache, models)
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation and guide
```

---

## Getting Started

### 1. Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13 installed on your machine.

### 2. Create and Activate a Virtual Environment

**Windows (PowerShell):**
```powershell
py -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Running the API

Start the development server with Uvicorn:

```bash
uvicorn app.main:app --reload
```

Once running, the API will be available at:
- **Root**: `http://127.0.0.1:8000/`
- **Interactive Swagger UI Docs**: `http://127.0.0.1:8000/docs`
- **Alternative ReDoc**: `http://127.0.0.1:8000/redoc`

---

## API Endpoints & Examples

### 1. Health Check
- **Endpoint**: `GET /api/health`
- **Description**: Verifies that the API is alive and the model is loaded in memory.

**cURL:**
```bash
curl -X GET http://127.0.0.1:8000/api/health
```

**Sample Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "cardiffnlp/twitter-roberta-base-sentiment-latest",
  "device": "cpu"
}
```

---

### 2. Analyze Single Text
- **Endpoint**: `POST /api/analyze`
- **Constraints**: 1 to 500 words.

**cURL:**
```bash
curl -X POST http://127.0.0.1:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "I am having an amazing time working on this project!"}'
```

**Sample Response:**
```json
{
  "text": "I am having an amazing time working on this project!",
  "sentiment": "positive",
  "confidence": 0.9845,
  "word_count": 10
}
```

---

### 3. Analyze Batch of Texts
- **Endpoint**: `POST /api/analyze/batch`
- **Constraints**: 1 to 10 texts (1 to 500 words per text).

**cURL:**
```bash
curl -X POST http://127.0.0.1:8000/api/analyze/batch \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "The product exceeded all my expectations!",
      "The package was delivered on Tuesday.",
      "Customer service was terrible and unhelpful."
    ]
  }'
```

**Sample Response:**
```json
{
  "total": 3,
  "results": [
    {
      "text": "The product exceeded all my expectations!",
      "sentiment": "positive",
      "confidence": 0.9812,
      "word_count": 6
    },
    {
      "text": "The package was delivered on Tuesday.",
      "sentiment": "neutral",
      "confidence": 0.8931,
      "word_count": 6
    },
    {
      "text": "Customer service was terrible and unhelpful.",
      "sentiment": "negative",
      "confidence": 0.9754,
      "word_count": 6
    }
  ]
}
```

---

## Running Automated Tests

Run the test suite using `pytest`:

```bash
pytest -v
```

This verifies:
- Service health endpoint
- Single text classification (`positive`, `negative`, `neutral`)
- Batch text classification
- Word count constraints (0 words, >500 words)
- Batch size constraints (>10 texts, empty list)
