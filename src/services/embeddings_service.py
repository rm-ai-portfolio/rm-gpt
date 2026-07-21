import logging
from typing import List
import numpy as np
from langchain_core.embeddings import Embeddings
from huggingface_hub import InferenceClient

logger = logging.getLogger(__name__)

class HFInferenceEmbeddings(Embeddings):
    def __init__(self, model: str, api_key: str, provider: str = "hf-inference"):
        self.model = model
        self.client = InferenceClient(provider=provider, api_key=api_key)

    def _extract(self, text: str) -> List[float]:
        try:
            output = self.client.feature_extraction(text, model=self.model)
        except Exception as e:
            logger.error(f"HF embedding error: {e}")
            raise

        arr = np.asarray(output, dtype=np.float32)

        if arr.ndim == 0:
            raise ValueError("Embedding returned scalar instead of vector")

        # Common shapes: (dim,), (1, dim), (seq_len, dim), (1, seq_len, dim)
        if arr.ndim == 2:
            if arr.shape[0] == 1:
                arr = arr.squeeze(0)
            else:
                arr = arr.mean(axis=0)
        elif arr.ndim > 2:
            arr = arr.mean(axis=tuple(range(arr.ndim - 1)))

        return arr.flatten().tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._extract(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._extract(text)
