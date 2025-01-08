from typing import Dict, Any
from abc import ABC, abstractmethod

class Factory(ABC):
    @staticmethod
    @abstractmethod
    def create(provider: str, config: Dict[str, Any]):
        pass

class LlmFactory(Factory):
    @staticmethod
    def create(provider: str, config: Dict[str, Any]):
        if provider == "openai":
            from .llms.openai import OpenAILLM
            return OpenAILLM(config)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

class EmbedderFactory(Factory):
    @staticmethod
    def create(provider: str, config: Dict[str, Any]):
        if provider == "openai":
            from .embeddings.openai import OpenAIEmbedder
            return OpenAIEmbedder(config)
        else:
            raise ValueError(f"Unsupported embedder provider: {provider}")

class VectorStoreFactory(Factory):
    @staticmethod
    def create(provider: str, config: Dict[str, Any]):
        if provider == "faiss":
            from .vector_stores.faiss import FaissVectorStore
            return FaissVectorStore(config)
        elif provider == "pinecone":
            from .vector_stores.pinecone import PineconeVectorStore
            return PineconeVectorStore(config)
        else:
            raise ValueError(f"Unsupported vector store provider: {provider}")
