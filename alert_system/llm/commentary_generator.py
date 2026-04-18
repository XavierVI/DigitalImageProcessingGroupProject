"""Generate driver alerts and commentary using LLM models."""

import torch
from typing import Optional, Dict


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
        tokenizer,
        model,
        device=None,
        max_new_tokens: int = 50,
        num_beams: int = 5,
        early_stopping: bool = True
    ):
        """Initialize the commentary generator.

        Args:
            tokenizer: Hugging Face tokenizer
            model: Seq2seq model (e.g., T5, BART)
            device: torch.device for inference
            max_new_tokens: Maximum length of generated text
            num_beams: Beam search parameter
            early_stopping: Whether to use early stopping
        """
        self.tokenizer = tokenizer
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self.early_stopping = early_stopping

    def generate(self, prompt: str, max_new_tokens: Optional[int] = None) -> str:
        """Generate commentary from a prompt.

        Args:
            prompt (str): Input prompt for the LLM
            max_new_tokens (int, optional): Override default max_new_tokens

        Returns:
            str: Generated commentary text
        """
        max_tokens = max_new_tokens or self.max_new_tokens

        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Generate text
        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                max_new_tokens=max_tokens,
                num_beams=self.num_beams,
                early_stopping=self.early_stopping
            )

        # Decode output
        commentary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return commentary

    def generate_alert(self, prompt: str, urgency: str = "normal") -> Dict[str, str]:
        """Generate an alert with urgency level.

        Args:
            prompt (str): Input prompt
            urgency (str): 'low', 'normal', or 'high'

        Returns:
            dict: Dictionary with 'alert' and 'urgency' keys
        """
        # Adjust parameters based on urgency
        if urgency == "high":
            max_tokens = min(30, self.max_new_tokens)
        elif urgency == "low":
            max_tokens = self.max_new_tokens
        else:
            max_tokens = self.max_new_tokens

        alert_prompt = f"[{urgency.upper()}] {prompt}"
        commentary = self.generate(alert_prompt, max_tokens)

        return {
            "alert": commentary,
            "urgency": urgency
        }
