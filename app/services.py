"""Sentiment analysis service using Hugging Face Transformers and PyTorch."""
import logging
from typing import Tuple, List, Optional
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from app.config import MODEL_NAME

logger = logging.getLogger(__name__)

# Fallback label map for 3-class sentiment models if config id2label is missing
DEFAULT_ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}


class SentimentAnalyzer:
    """Manages the Hugging Face model and tokenizer lifecycle and inference."""

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self.model_name: str = model_name
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModelForSequenceClassification] = None
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self._is_loaded: bool = False

    def load_model(self) -> None:
        """Download (if not cached) and load model and tokenizer into memory."""
        if self._is_loaded:
            return

        logger.info(f"Loading tokenizer and model: {self.model_name} on device: {self.device}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            self._is_loaded = True
            logger.info("Sentiment model successfully loaded.")
        except Exception as exc:
            logger.error(f"Failed to load model {self.model_name}: {exc}")
            raise RuntimeError(f"Could not load Hugging Face model: {exc}") from exc

    @property
    def is_loaded(self) -> bool:
        """Check if model is currently ready for inference."""
        return self._is_loaded and self.model is not None and self.tokenizer is not None

    def _format_label(self, label_idx: int) -> str:
        """Retrieve standardized label (positive, negative, neutral) from model config."""
        if self.model and hasattr(self.model.config, "id2label") and self.model.config.id2label:
            raw_label = str(self.model.config.id2label.get(label_idx, "")).lower()
            if raw_label in ("positive", "negative", "neutral"):
                return raw_label
            # Handle common variations e.g. "pos", "neg", "neu", "label_0"
            if "pos" in raw_label:
                return "positive"
            if "neg" in raw_label:
                return "negative"
            if "neu" in raw_label:
                return "neutral"
        return DEFAULT_ID2LABEL.get(label_idx, "neutral")

    def predict(self, text: str) -> Tuple[str, float]:
        """Perform sentiment analysis on a single text string."""
        results = self.predict_batch([text])
        return results[0]

    def predict_batch(self, texts: List[str]) -> List[Tuple[str, float]]:
        """Perform batched sentiment analysis for enhanced throughput."""
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded. Please wait for service initialization.")

        # Tokenize batch with padding and truncation (max roberta sequence length is 512)
        encoded_inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        encoded_inputs = {k: v.to(self.device) for k, v in encoded_inputs.items()}

        with torch.no_grad():
            outputs = self.model(**encoded_inputs)
            # Apply softmax over logits to obtain normalized confidence probabilities
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            confidences, predicted_indices = torch.max(probabilities, dim=-1)

        results: List[Tuple[str, float]] = []
        for idx in range(len(texts)):
            label_idx = int(predicted_indices[idx].item())
            confidence_score = round(float(confidences[idx].item()), 4)
            label = self._format_label(label_idx)
            results.append((label, confidence_score))

        return results


# Global singleton instance of the sentiment analyzer
sentiment_service = SentimentAnalyzer()
