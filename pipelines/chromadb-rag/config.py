import os
import sys

from enum import Enum
from dataclasses import dataclass
from typing import Any, Optional
from dotenv import load_dotenv

load_dotenv()


class EmbeddingProvider(Enum):
    OLLAMA = "ollama"
    SENTENCE_TRANSFORMERS = "sentence-transformers"


@dataclass
class OllamaConfig:
    server_url: str
    model: str
    embedding_model: str
    timeout: int = 120

    def validate(self) -> list[str]:
        errors = []

        if not self.server_url:
            errors.append("OLLAMA_SERVER_URL is not set")
        elif not self.server_url.startswith(("http://", "https://")):
            errors.append(
                f"OLLAMA_SERVER_URL must start with http:// or https://, got: {self.server_url}"
            )

        if not self.model:
            errors.append("OLLAMA_MODEL is not set")

        if not self.embedding_model:
            errors.append("OLLAMA_EMBEDDING_MODEL is not set")

        if self.timeout <= 0:
            errors.append(f"Ollama timeout must be positive, got: {self.timeout}")

        return errors


@dataclass
class SentenceTransformersConfig:
    model: str

    def validate(self) -> list[str]:
        errors = []

        if not self.model:
            errors.append("SENTENCE_TRANSFORMERS_MODEL is not set")

        return errors


@dataclass
class ChromaDBConfig:
    persist_path: str
    collection_name: str

    def validate(self) -> list[str]:
        errors = []

        if not self.persist_path:
            errors.append("CHROMA_PERSIST_PATH is not set")

        if not self.collection_name:
            errors.append("CHROMA_COLLECTION_NAME is not set")
        elif len(self.collection_name) > 63:
            errors.append(
                f"CHROMA_COLLECTION_NAME too long (max 63 chars): {self.collection_name}"
            )
        elif not self.collection_name.replace("-", "").replace("_", "").isalnum():
            errors.append(
                f"CHROMA_COLLECTION_NAME can only contain alphanumeric, dash, and underscore: {self.collection_name}"
            )

        return errors


@dataclass
class PipelineConfig:
    top_k: int
    chunk_size: int
    chunk_overlap: int

    def validate(self) -> list[str]:
        errors = []

        if self.top_k <= 0:
            errors.append(f"TOP_K must be positive, got: {self.top_k}")
        if self.top_k > 100:
            errors.append(f"TOP_K too large (max 100), got: {self.top_k}")

        if self.chunk_size <= 0:
            errors.append(f"CHUNK_SIZE must be positive, got: {self.chunk_size}")
        if self.chunk_size > 50:
            errors.append(f"CHUNK_SIZE too large (max 50), got: {self.chunk_size}")

        if self.chunk_overlap < 0:
            errors.append(
                f"CHUNK_OVERLAP must be non-negative, got: {self.chunk_overlap}"
            )
        if self.chunk_overlap >= self.chunk_size:
            errors.append(
                f"CHUNK_OVERLAP ({self.chunk_overlap}) must be less than CHUNK_SIZE ({self.chunk_size})"
            )

        return errors


@dataclass
class LLMConfig:
    num_predict: int = 512
    temperature: float = 0.2
    num_ctx: int = 2048
    top_p: float = 0.9

    def validate(self) -> list[str]:
        errors = []

        if self.num_predict <= 0:
            errors.append(f"num_predict must be positive, got: {self.num_predict}")

        if not 0.0 <= self.temperature <= 2.0:
            errors.append(
                f"temperature must be between 0.0 and 2.0, got: {self.temperature}"
            )

        if self.num_ctx <= 0:
            errors.append(f"num_ctx must be positive, got: {self.num_ctx}")

        if not 0.0 <= self.top_p <= 1.0:
            errors.append(f"top_p must be between 0.0 and 1.0, got: {self.top_p}")

        return errors


class Config:
    def __init__(self):
        self.embedding_provider = self._get_embedding_provider()

        self.ollama = OllamaConfig(
            server_url=os.getenv("OLLAMA_SERVER_URL", "http://ollama:11434"),
            model=os.getenv("OLLAMA_MODEL", "llama3.2"),
            embedding_model=os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"),
            timeout=self._get_int_env("OLLAMA_TIMEOUT", 120),
        )

        self.sentence_transformers = SentenceTransformersConfig(
            model=os.getenv(
                "SENTENCE_TRANSFORMERS_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
            )
        )

        self.chromadb = ChromaDBConfig(
            persist_path=os.getenv("CHROMA_PERSIST_PATH", "/data/chroma_db"),
            collection_name=os.getenv("CHROMA_COLLECTION_NAME", "document_collection"),
        )

        self.pipeline = PipelineConfig(
            top_k=self._get_int_env("TOP_K", 4),
            chunk_size=self._get_int_env("CHUNK_SIZE", 3),
            chunk_overlap=self._get_int_env("CHUNK_OVERLAP", 1),
        )

        self.llm = LLMConfig(
            num_predict=self._get_int_env("LLM_NUM_PREDICT", 512),
            temperature=self._get_float_env("LLM_TEMPERATURE", 0.2),
            num_ctx=self._get_int_env("LLM_NUM_CTX", 2048),
            top_p=self._get_float_env("LLM_TOP_P", 0.9),
        )

    def _get_embedding_provider(self) -> EmbeddingProvider:
        provider = os.getenv("EMBEDDING_PROVIDER", "ollama").lower()

        try:
            return EmbeddingProvider(provider)
        except ValueError:
            valid_providers = [p.value for p in EmbeddingProvider]
            raise ValueError(
                f"Invalid EMBEDDING_PROVIDER: '{provider}'. "
                f"Must be one of: {valid_providers}"
            )

    def _get_int_env(self, key: str, default: int) -> int:
        value = os.getenv(key)
        if value is None:
            return default

        try:
            return int(value)
        except ValueError:
            raise ValueError(f"{key} must be an integer, got: {value}")

    def _get_float_env(self, key: str, default: float) -> float:
        value = os.getenv(key)
        if value is None:
            return default

        try:
            return float(value)
        except ValueError:
            raise ValueError(f"{key} must be a float, got: {value}")

    def validate(self) -> tuple[bool, list[str]]:
        """
        Validate all configuration

        Returns:
            tuple: (is_valid, list_of_errors)
        """
        all_errors = []

        all_errors.extend(self.ollama.validate())

        if self.embedding_provider == EmbeddingProvider.OLLAMA:
            pass
        else:
            all_errors.extend(self.sentence_transformers.validate())

        all_errors.extend(self.chromadb.validate())

        all_errors.extend(self.pipeline.validate())

        all_errors.extend(self.llm.validate())

        return len(all_errors) == 0, all_errors

    def get_summary(self) -> dict[str, Any]:
        return {
            "embedding_provider": self.embedding_provider.value,
            "ollama": {
                "server_url": self.ollama.server_url,
                "model": self.ollama.model,
                "embedding_model": self.ollama.embedding_model,
                "timeout": self.ollama.timeout,
            },
            "sentence_transformers": (
                {
                    "model": self.sentence_transformers.model,
                }
                if self.embedding_provider == EmbeddingProvider.SENTENCE_TRANSFORMERS
                else None
            ),
            "chromadb": {
                "persist_path": self.chromadb.persist_path,
                "collection_name": self.chromadb.collection_name,
            },
            "pipeline": {
                "top_k": self.pipeline.top_k,
                "chunk_size": self.pipeline.chunk_size,
                "chunk_overlap": self.pipeline.chunk_overlap,
            },
            "llm": {
                "num_predict": self.llm.num_predict,
                "temperature": self.llm.temperature,
                "num_ctx": self.llm.num_ctx,
                "top_p": self.llm.top_p,
            },
        }

    def log_summary(self, logger):
        logger.info("=" * 70)
        logger.info("Configuration Summary")
        logger.info("=" * 70)

        logger.info(f"Embedding Provider: {self.embedding_provider.value}")
        logger.info("")

        logger.info("Ollama Configuration:")
        logger.info(f"  Server URL: {self.ollama.server_url}")
        logger.info(f"  LLM Model: {self.ollama.model}")
        logger.info(f"  Embedding Model: {self.ollama.embedding_model}")
        logger.info(f"  Timeout: {self.ollama.timeout}s")
        logger.info("")

        if self.embedding_provider == EmbeddingProvider.SENTENCE_TRANSFORMERS:
            logger.info("Sentence Transformers Configuration: ")
            logger.info(f"  Model: {self.sentence_transformers.model}")
            logger.info("")

        logger.info("ChromaDB Configuration:")
        logger.info(f"  Persist Path: {self.chromadb.persist_path}")
        logger.info(f"  Collection: {self.chromadb.collection_name}")
        logger.info("")

        logger.info("Pipeline Configuration:")
        logger.info(f"  Top K: {self.pipeline.top_k}")
        logger.info(f"  Chunk Size: {self.pipeline.chunk_size}")
        logger.info(f"  Chunk Overlap: {self.pipeline.chunk_overlap}")
        logger.info("")

        logger.info("LLM Generation Configuration:")
        logger.info(f"  Max Tokens: {self.llm.num_predict}")
        logger.info(f"  Temperature: {self.llm.temperature}")
        logger.info(f"  Context Window: {self.llm.num_ctx}")
        logger.info(f"  Top P: {self.llm.top_p}")
        logger.info("=" * 70)


_config: Optional[Config] = None


def get_config() -> Config:
    """
    Get or create the configuration singleton

    Returns:
        Config: The application configuration
    """
    global _config

    if _config is None:
        _config = Config()

        is_valid, errors = _config.validate()

        if not is_valid:
            print("=" * 70, file=sys.stderr)
            print("CONFIGURATION ERRORS", file=sys.stderr)
            print("=" * 70, file=sys.stderr)
            for error in errors:
                print(f"  ✗ {error}", file=sys.stderr)
            print("=" * 70, file=sys.stderr)
            raise ValueError(
                f"Configuration validation failed with {len(errors)} error(s)"
            )

    return _config


def validate_config() -> tuple[bool, list[str]]:
    """
    Validate configuration without creating singleton

    Returns:
        tuple: (is_valid, list_or_errors)
    """
    config = Config()
    return config.validate()


if __name__ == "__main__":
    """Run configuration validation"""
    print("=" * 70)
    print("Configuration Validation Tool")
    print("=" * 70)
    print()

    try:
        config = get_config()
        print("✓ Configuration is valid!")
        print()
        config.log_summary(type("Logger", (), {"info": print})())

    except ValueError as e:
        print("✗ Configuration validation failed:")
        print(f"  {e}")
        sys.exit(1)

    except Exception as e:
        print("✗ Unexpected error:")
        print(f"  {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
