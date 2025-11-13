import logging
from typing import Sequence

import onnxruntime as ort
from tokenizers import Tokenizer
from numpy import ndarray, argmax

_session = None
_tokenizer = None

logger = logging.getLogger(__name__)


def start_session(model_path: str) -> None:
    global _session
    _session = ort.InferenceSession(
        model_path,
        sess_options=ort.SessionOptions(),
        providers=["CPUExecutionProvider"],
    )


def start_tokenizer(tokenizer_path: str) -> None:
    global _tokenizer
    _tokenizer = Tokenizer.from_file(tokenizer_path)


def predict(input_data: str) -> int:
    """Function to run prediction entirely within worker process."""
    global _session

    try:
        if _tokenizer is None:
            raise Exception("Tokenizer not initialized in worker")
        if _session is None:
            raise Exception("Session not initialized in worker")

        tokens = _tokenizer.encode(input_data).ids

        # Tokenize input
        attention_mask = get_attention_mask(tokens)

        # Prepare input for ONNX model
        model_input = {
            "input_ids": [tokens],
            "attention_mask": [attention_mask],
        }

        # Run inference
        result = _session.run(None, model_input)

        if not isinstance(result[0], ndarray):
            return 0

        first_hidden_state = result[0].tolist()[0]
        label = argmax(first_hidden_state).item()

        return label

    except Exception as e:
        logger.error(f"Error in worker prediction: {e}")
        return 0


def get_attention_mask(input_ids: Sequence, pad_token_id: int = 0) -> list:
    return [1 if token != pad_token_id else 0 for token in input_ids]
