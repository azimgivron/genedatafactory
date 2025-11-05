from typing import List

import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "dmis-lab/biobert-base-cased-v1.1"


def _pick_device() -> str:
    """Prefer Apple Silicon GPU (MPS), then CUDA, else CPU.

    Returns:
        str: Hardware to use.
    """
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class BioBERTMeanEncoder:
    """Encodes biomedical text into dense vector embeddings using BioBERT.

    This class loads the pre-trained BioBERT model from Hugging Face and applies
    mean pooling over token embeddings to generate a single 768-dimensional
    vector per input text. The embeddings can be used for downstream tasks such as
    gene similarity analysis, clustering, or functional representation.

    Attributes:
        device (str): The device on which computations are performed
            ('cuda' if available, otherwise 'cpu').
        tokenizer (AutoTokenizer): Hugging Face tokenizer for the BioBERT model.
        model (AutoModel): Pre-trained BioBERT model loaded for inference.
    """

    def __init__(self):
        """Initializes the BioBERTMeanEncoder."""
        self.device = _pick_device()
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModel.from_pretrained(MODEL_NAME).to(self.device)
        self.model.eval()

    @staticmethod
    def _mean_pool(
        last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Applies mean pooling across token embeddings while ignoring padding.

        Args:
            last_hidden_state (torch.Tensor): Tensor of shape
                (batch_size, sequence_length, hidden_size) containing token embeddings.
            attention_mask (torch.Tensor): Tensor of shape
                (batch_size, sequence_length) indicating non-padding tokens (1 = token, 0 = pad).

        Returns:
            torch.Tensor: Tensor of shape (batch_size, hidden_size) representing
            mean-pooled embeddings for each input sequence.
        """
        mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
        summed = (last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    @torch.inference_mode()
    def encode(self, texts: List[str]) -> torch.Tensor:
        """Encodes input biomedical text into BioBERT embeddings using mean pooling.

        Args:
            texts (List[str]): List of input texts to encode. Each text should be
                ≤ 512 tokens after tokenization (longer texts will be truncated).

        Returns:
            torch.Tensor: A tensor of shape (num_texts, 768) containing BioBERT
            mean-pooled embeddings for each input text.
        """
        embs = []
        if self.device == "mps":
            batch_size = 128
        elif self.device == "cuda":
            batch_size = 64
        else:
            batch_size = 16
        for i in tqdm(range(0, len(texts), batch_size), desc="BioBERT encoding"):
            batch = texts[i : i + batch_size]
            toks = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)

            out = self.model(**toks)
            emb = self._mean_pool(out.last_hidden_state, toks.attention_mask)
            embs.append(emb.cpu())

        return torch.cat(embs, dim=0)
