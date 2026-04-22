"""Generate driver alerts and commentary using LLM models."""

import torch
from typing import Optional, Dict

from transformers import T5Tokenizer, T5ForConditionalGeneration

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
        hugging_face_model: str = "google/flan-t5-small",
        device=None,
        max_new_tokens: int = 50,
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
        
        if hugging_face_model == "google/flan-t5-small":
            self.tokenizer = T5Tokenizer.from_pretrained(
                hugging_face_model,
                cache_dir=os.path.join(os.getcwd(), "models")
            )
            self.model = T5ForConditionalGeneration.from_pretrained(
                hugging_face_model,
                cache_dir=os.path.join(os.getcwd(), "models")
            ).to(device)

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self.early_stopping = early_stopping

    def generate(self, prompt: str) -> str:
        """Generate commentary from a prompt.

        Args:
            prompt (str): Input prompt for the LLM
            max_new_tokens (int, optional): Override default max_new_tokens

        Returns:
            str: Generated commentary text
        """
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Generate text
        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                max_new_tokens=self.max_new_tokens,
                num_beams=self.num_beams,
                early_stopping=self.early_stopping
            )

        # Decode output
        commentary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return commentary

