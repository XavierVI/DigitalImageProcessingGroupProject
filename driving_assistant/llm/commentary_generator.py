"""Generate driver alerts and commentary using LLM models."""

import torch
from typing import Optional

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import os

class CommentaryGenerator:
    """Generates natural language commentary from prompts using a pre-trained LLM.

    Args:
        tokenizer: Hugging Face tokenizer for the LLM
        model: Pre-trained seq2seq model for text generation
        device: Device to run inference on (cuda or cpu)
        max_new_tokens: Maximum tokens to generate
        num_beams: Number of beams for beam search
        early_stopping: Whether to stop when beam search has found good solutions
    """

    def __init__(
        self,
        hugging_face_model: str = "google/flan-t5-base",
        device=None,
        max_new_tokens: int = 80,
        num_beams: int = 5,
        early_stopping: bool = True
    ):
        """Initialize the commentary generator.

        Args:
            hugging_face_model: Path or name of the Hugging Face model
            device: torch.device for inference
            max_new_tokens: Maximum length of generated text
            num_beams: Beam search parameter
            early_stopping: Whether to use early stopping
        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            hugging_face_model,
            cache_dir=os.path.join(os.getcwd(), "models")
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            hugging_face_model,
            cache_dir=os.path.join(os.getcwd(), "models")
        )

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self.early_stopping = early_stopping

        # Keep prompt length within model context window to avoid runtime warnings/errors.
        self.input_max_length = min(
            getattr(self.tokenizer, "model_max_length", 512),
            getattr(self.model.config, "n_positions", 512),
            512,
        )

    def generate(self, prompt: str) -> str:
        """Generate commentary from a prompt.

        Args:
            prompt (str): Input prompt for the LLM
            max_new_tokens (int, optional): Override default max_new_tokens

        Returns:
            str: Generated commentary text
        """
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.input_max_length,
        ).to(self.device)

        # Generate text
        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                max_new_tokens=self.max_new_tokens,
                num_beams=self.num_beams,
                early_stopping=self.early_stopping,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
            )

        # Decode output
        commentary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return commentary

