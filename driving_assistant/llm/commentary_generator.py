"""Generate driver alerts and commentary using LLM models."""

import torch
from typing import Optional, Dict

from transformers import (
    T5Tokenizer, T5ForConditionalGeneration,
    AutoTokenizer, AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    BitsAndBytesConfig
)

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
        hugging_face_model: str = "microsoft/Phi-3-mini-4k-instruct",
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

        if hugging_face_model in ("google/flan-t5-small", "google/flan-t5-large"):
            self.tokenizer = AutoTokenizer.from_pretrained(
                hugging_face_model,
                cache_dir=os.path.join(os.getcwd(), "models")
            )
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                hugging_face_model,
                cache_dir=os.path.join(os.getcwd(), "models")
            )
            self.commentary_method = self._t5_generate

        elif hugging_face_model == "microsoft/Phi-3-mini-4k-instruct": #Phi - 3 - mini
            self.tokenizer = AutoTokenizer.from_pretrained(
                hugging_face_model,
                cache_dir=os.path.join(os.getcwd(), "models"),
                trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(
                hugging_face_model,
                cache_dir=os.path.join(os.getcwd(), "models"),
                torch_dtype=torch.float16,
                trust_remote_code=True,
                attn_implementation = "eager"
            )
            self.commentary_method = self.llm_generate

        elif hugging_face_model == "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B":
            self.tokenizer = AutoTokenizer.from_pretrained(
                hugging_face_model,
                cache_dir=os.path.join(os.getcwd(), "models")
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = AutoModelForCausalLM.from_pretrained(
                hugging_face_model,
                cache_dir=os.path.join(os.getcwd(), "models"),
            )
            self.commentary_method = self.llm_generate

        elif hugging_face_model == "deepseek-ai/DeepSeek-V3.2":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                hugging_face_model,
                cache_dir=os.path.join(os.getcwd(), "models"),
                trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(hugging_face_model,
                cache_dir=os.path.join(os.getcwd(), "models"),
                quantization_config=quant_config,
                device_map="auto",
                trust_remote_code=True
            )

            self.commentary_method = self.llm_generate

        else:
            raise ValueError(
                f"Unsupported model: {hugging_face_model}. "
                "Choose from: google/flan-t5-small, google/flan-t5-large, "
                "microsoft/Phi-3-mini-4k-instruct, "
                "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B, "
                "deepseek-ai/DeepSeek-V3.2"
            )


        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()  # Set model to evaluation mode
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
        Returns:
            str: Generated commentary text
        """
        return self.commentary_method(prompt)

    def llm_generate(self, prompt: str) -> str:
        messages = [
            {
                "role": "system",
                "content": "You are an expert driving assistant. Analyze the detected objects from dashcam footage and warn the driver about any immediate risks. Be specific about what the object is, where it is, and what action to take."
            },
            {"role": "user", "content": prompt}
        ]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict = True
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def _t5_generate(self, prompt: str) -> str:
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
                no_repeat_ngram_size=3
            )

        # Decode output
        commentary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return commentary

