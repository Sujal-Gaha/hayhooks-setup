import os
import time
import tempfile


from pathlib import Path
from typing import Any, Generator, Optional, Union

from fastapi import UploadFile

from haystack import Document, Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)
from haystack.components.converters import PyPDFToDocument, TextFileToDocument
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.writers import DocumentWriter

from hayhooks import BasePipelineWrapper, streaming_generator
from hayhooks.server.logger import log

from haystack_integrations.components.generators.ollama import OllamaGenerator
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever

from haystack_integrations.components.embedders.ollama import (
    OllamaDocumentEmbedder,
    OllamaTextEmbedder,
)

from .config import get_config, EmbeddingProvider


class PipelineWrapper(BasePipelineWrapper):

    def setup(self):
        start_time = time.time()
        log.info("Starting pipeline setup...")

        try:
            self.config = get_config()
            self.config.log_summary(log)

            os.makedirs(self.config.chromadb.persist_path, exist_ok=True)
            log.info(f"ChromaDB directory ensured: {self.config.chromadb.persist_path}")

            self.document_store = ChromaDocumentStore(
                collection_name=self.config.chromadb.collection_name,
                persist_path=self.config.chromadb.persist_path,
            )
            log.info("ChromaDB initialized")

            self._initialize_embedders()

            self._setup_indexing_pipeline()
            self._setup_query_pipeline()

            elapsed = time.time() - start_time
            log.info("=" * 70)
            log.info(f"Pipeline setup completed in {elapsed:.2f}s")
            log.info("=" * 70)

        except Exception as e:
            log.error(f"Failed to setup pipeline: {str(e)}")
            raise

    def _initialize_embedders(self):
        if self.config.embedding_provider == EmbeddingProvider.OLLAMA:
            log.info("Initializing Ollama embedders")
            log.info(f"  Model: {self.config.ollama.embedding_model}")
            log.info(f"  Server: {self.config.ollama.server_url}")

            self.doc_embedder = OllamaDocumentEmbedder(
                model=self.config.ollama.embedding_model,
                url=self.config.ollama.server_url,
                timeout=self.config.ollama.timeout,
            )

            self.text_embedder = OllamaTextEmbedder(
                model=self.config.ollama.embedding_model,
                url=self.config.ollama.server_url,
                timeout=self.config.ollama.timeout,
            )

            log.info("Ollama embedders initialized")

        else:
            log.info("Initializing Sentence Transformers")
            log.info(f"  Model: {self.config.sentence_transformers.model}")

            self.doc_embedder = SentenceTransformersDocumentEmbedder(
                model=self.config.sentence_transformers.model
            )
            self.doc_embedder.warm_up()

            self.text_embedder = SentenceTransformersTextEmbedder(
                model=self.config.sentence_transformers.model
            )
            self.text_embedder.warm_up()

            log.info("Sentence Transformers embedders loaded and warmed up")

    def _setup_indexing_pipeline(self):
        self.indexing_pipeline = Pipeline()

        pdf_converter = PyPDFToDocument()
        txt_converter = TextFileToDocument()

        splitter = DocumentSplitter(
            split_by="sentence",
            split_length=self.config.pipeline.chunk_size,
            split_overlap=self.config.pipeline.chunk_overlap,
        )

        writer = DocumentWriter(document_store=self.document_store)

        self.indexing_pipeline.add_component("pdf_converter", pdf_converter)
        self.indexing_pipeline.add_component("txt_converter", txt_converter)
        self.indexing_pipeline.add_component("splitter", splitter)
        self.indexing_pipeline.add_component("embedder", self.doc_embedder)
        self.indexing_pipeline.add_component("writer", writer)

        log.info("Indexing pipeline initialized")

    def _setup_query_pipeline(self):
        self.pipeline = Pipeline()

        retriever = ChromaEmbeddingRetriever(
            document_store=self.document_store, top_k=self.config.pipeline.top_k
        )

        template = """
        You are a helpful assistant that answers questions based on the provided context.

        Context:
        {% for document in documents %}
        {{ document.content }}
        {% endfor %}

        Question: {{ question }}

        Please provide a clear and concise answer based on the context above. If the context doesn't contain enough information to answer the question, say so.

        Answer:
        """

        prompt_builder = PromptBuilder(
            template=template, required_variables=["documents", "question"]
        )

        ollama_generator = OllamaGenerator(
            model=self.config.ollama.model,
            url=self.config.ollama.server_url,
            timeout=self.config.ollama.timeout,
            generation_kwargs={
                "num_predict": self.config.llm.num_predict,
                "temperature": self.config.llm.temperature,
                "num_ctx": self.config.llm.num_ctx,
                "top_p": self.config.llm.top_p,
            },
        )

        self.pipeline.add_component("text_embedder", self.text_embedder)
        self.pipeline.add_component("retriever", retriever)
        self.pipeline.add_component("prompt_builder", prompt_builder)
        self.pipeline.add_component("llm", ollama_generator)

        self.pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
        self.pipeline.connect("retriever.documents", "prompt_builder.documents")
        self.pipeline.connect("prompt_builder", "llm")

        log.info("Query pipeline initialized")

    def _get_file_extension(self, filename: str) -> str:
        return Path(filename).suffix.lower()

    def _process_uploaded_file(self, file: UploadFile) -> list[Document]:
        if not file.filename:
            raise ValueError("File must have a filename")

        log.info(f"Processing file: {file.filename}")

        suffix = self._get_file_extension(file.filename)

        with tempfile.NamedTemporaryFile(
            delete=False, suffix=suffix, mode="wb"
        ) as tmp_file:
            content = file.file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name

        try:
            if suffix == ".pdf":
                result = self.indexing_pipeline.run(
                    {"pdf_converter": {"sources": [tmp_path]}}
                )
                documents = result["pdf_converter"]["documents"]
            elif suffix == ".txt":
                result = self.indexing_pipeline.run(
                    {"txt_converter": {"sources": [tmp_path]}}
                )
                documents = result["txt_converter"]["documents"]
            else:
                raise ValueError(
                    f"Unsupported file type: {suffix}. " f"Supported types: .pdf, .txt"
                )

            for doc in documents:
                doc.meta["filename"] = file.filename
                doc.meta["upload_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
                doc.meta["file_type"] = suffix

            log.info(f"Extracted {len(documents)} document(s) from {file.filename}")
            return documents

        except Exception as e:
            log.error(f"Error processing {file.filename}: {str(e)}")
            raise
        finally:
            try:
                os.unlink(tmp_path)
            except Exception as e:
                log.warning(f"Could not delete temp file {tmp_path}: {e}")

    def index_files(self, files: list[UploadFile]) -> dict[str, Any]:
        if not files:
            raise ValueError("No files provided for indexing")

        log.info("=" * 70)
        log.info(f"Starting indexing for {len(files)} file(s)")
        log.info("=" * 70)
        start = time.time()

        all_documents = []
        processed_files = []
        errors = []

        for idx, file in enumerate(files, 1):
            try:
                log.info(f"[{idx}/{len(files)}] Processing: {file.filename}")
                documents = self._process_uploaded_file(file)
                all_documents.extend(documents)
                processed_files.append(file.filename)
            except Exception as e:
                log.error(f"[{idx}/{len(files)}] Failed: {file.filename} - {str(e)}")
                errors.append({"filename": file.filename, "error": str(e)})

        chunks_created = 0

        if all_documents:
            try:
                log.info(f"Splitting {len(all_documents)} documents to chunks...")
                split_result = self.indexing_pipeline.run(
                    {"splitter": {"documents": all_documents}}
                )
                split_docs = split_result["splitter"]["documents"]
                log.info(f"Created {len(split_docs)} chunks")

                log.info(
                    f"Generating embeddings using {self.config.embedding_provider.value}..."
                )
                embed_result = self.indexing_pipeline.run(
                    {"embedder": {"documents": split_docs}}
                )
                embedded_docs = embed_result["embedder"]["documents"]
                log.info(f"Generated embeddings for {len(embedded_docs)}")

                log.info("Writing to ChromaDB...")
                self.indexing_pipeline.run({"writer": {"documents": embedded_docs}})
                chunks_created = len(embedded_docs)
                log.info(f"Successfully indexed {chunks_created} chunks")

            except Exception as e:
                log.error(f"Error during indexing: {str(e)}")
                errors.append({"stage": "indexing", "error": str(e)})

        elapsed = time.time() - start

        log.info("=" * 70)
        log.info(f"Indexing completed in {elapsed:.2f}s")
        log.info(f"Files processed: {len(processed_files)}/{len(files)}")
        log.info(f"Chunks created: {chunks_created}")
        log.info(f"Errors: {len(errors)}")
        log.info("=" * 70)

        embedding_model = (
            self.config.ollama.embedding_model
            if self.config.embedding_provider == EmbeddingProvider.OLLAMA
            else self.config.sentence_transformers.model
        )

        return {
            "status": "success" if processed_files else "failed",
            "files_processed": len(processed_files),
            "total_files": len(files),
            "filenames": processed_files,
            "chunks_created": chunks_created,
            "embedding_provider": self.config.embedding_provider.value,
            "embedding_model": embedding_model,
            "errors": errors,
            "elapsed_time": f"{elapsed:.2f}s",
        }

    def run_api(
        self, files: Optional[list[UploadFile]] = None, question: str = ""
    ) -> dict[str, Any]:
        log.info("=" * 70)

        response = {}

        if files and len(files) > 0:
            try:
                index_result = self.index_files(files)
                response["indexing"] = index_result

                if not question:
                    return response

            except Exception as e:
                log.error(f"Indexing failed: {str(e)}")
                return {"error": "Indexing failed", "details": str(e)}

        if question:
            try:
                start = time.time()
                log.info(f"Running query: '{question[:100]}'")

                result = self.pipeline.run(
                    {
                        "text_embedder": {"text": question},
                        "prompt_builder": {"question": question},
                    }
                )

                reply = result["llm"]["replies"][0]

                retrieved_docs = result.get("retriever", {}).get("documents", [])

                elapsed = time.time() - start
                log.info(f"Query completed in {elapsed:.2f}s")

                response.update(
                    {
                        "reply": reply,
                        "elapsed_time": f"{elapsed:.2f}s",
                        "retrieved_documents": len(retrieved_docs),
                        "embedding_provider": self.config.embedding_provider.value,
                        "sources": [
                            {
                                "filename": doc.meta.get("filename", "unknown"),
                                "score": doc.score if hasattr(doc, "score") else None,
                            }
                            for doc in retrieved_docs[:3]
                        ],
                    }
                )

                return response

            except Exception as e:
                log.error(f"Query failed: {str(e)}")
                import traceback

                traceback.print_exc()
                return {"error": "Query failed", "details": str(e)}

        raise ValueError("Either 'files' or 'question' must be provided")

    def run_chat_completion(
        self, model: str, messages: list[dict], body: dict
    ) -> Union[str, Generator]:
        log.info(f"Chat completion result | model: {model}")

        question = ""
        for msg in reversed(messages):
            if msg.get("role") == "user" and msg.get("content"):
                question = str(msg["content"]).strip()
                break

        if not question:
            log.warning("No user message found in chat history")
            return "No question provided."

        top_k = int(body.get("top_k", self.config.pipeline.top_k))

        log.info(f"Streaming RAG | question: '{question[:80]}...' | top_k: {top_k}")

        # self.pipeline.get_component("retriever").top_k = top_k

        return streaming_generator(
            pipeline=self.pipeline,
            pipeline_run_args={
                "text_embedder": {"text": question},
                "prompt_builder": {"question": question},
            },
        )
