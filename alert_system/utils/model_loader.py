"""Load pre-trained models for object detection and LLM inference."""

import torch
from transformers import (
    DetrImageProcessor,
    DetrForObjectDetection,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)


def load_detection_model(model_name="facebook/detr-resnet-50", device=None):
    """Load pre-trained DETR object detection model.

    Args:
        model_name (str): Hugging Face model identifier for DETR
        device (torch.device, optional): Device to load model on. If None, auto-detects.

    Returns:
        tuple: (image_processor, model) for object detection
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading DETR model from {model_name}...")
    processor = DetrImageProcessor.from_pretrained(model_name)
    model = DetrForObjectDetection.from_pretrained(model_name).to(device)
    print(f"DETR model loaded on {device}")

    return processor, model


def load_llm_model(model_name="google/flan-t5-small", device=None):
    """Load pre-trained Language Model for text generation.

    Args:
        model_name (str): Hugging Face model identifier for the LLM
        device (torch.device, optional): Device to load model on. If None, auto-detects.

    Returns:
        tuple: (tokenizer, model) for text generation
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading LLM model from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    print(f"LLM model loaded on {device}")

    return tokenizer, model
