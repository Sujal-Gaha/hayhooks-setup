import os
import time

from typing import Any, Generator, Union, cast

from datasets import load_dataset

from haystack import Document, Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy
from haystack.components.retrievers import InMemoryBM25Retriever
from haystack.components.builders import PromptBuilder

from haystack_integrations.components.generators.ollama import OllamaGenerator

from hayhooks.server.logger import log
from hayhooks import BasePipelineWrapper, streaming_generator

from dotenv import load_dotenv


load_dotenv()

OLLAMA_SERVER_URL = os.getenv("OLLAMA_SERVER_URL") or ""
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL") or ""

TOP_K = 4

log.info("Pipeline module loaded")
log.info(f"OLLAMA_SERVER_URL resolved to: '{OLLAMA_SERVER_URL}'")
log.info(f"OLLAMA_MODEL resolved to: '{OLLAMA_MODEL}'")
log.info(f"Default TOP_K set to: {TOP_K}")

if not OLLAMA_SERVER_URL:
    log.warning("OLLAMA_SERVER_URL is empty – OllamaGenerator will likely fail")

if not OLLAMA_MODEL:
    log.warning("OLLAMA_MODEL is empty - Ollama Generator will likely fail")


class PipelineWrapper(BasePipelineWrapper):

    def setup(self):
        start_time = time.time()
        log.info("Pipeline setup started")

        log.info("Loading HotpotQA dataset (train[:5000])")
        dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split="train[:5000]")
        log.info(f"Dataset loaded with {len(dataset)} samples")

        docs = []
        log.info("Building Haystack documents")

        for idx, raw_doc in enumerate(dataset):
            doc = cast(dict[str, Any], raw_doc)
            titles = doc["context"]["title"]
            sentences_list = doc["context"]["sentences"]

            for title, sentences in zip(titles, sentences_list):
                content = f"{title}: {' '.join(sentences)}"

                docs.append(
                    Document(
                        content=content,
                        meta={
                            "title": title,
                            "source_question": doc["question"],
                            "source_answer": doc["answer"],
                        },
                    )
                )

            if idx > 0 and idx % 500 == 0:
                log.debug(f"Processed {idx} dataset entries")

        log.info(f"Constructed {len(docs)} documents")

        document_store = InMemoryDocumentStore()
        document_store.write_documents(docs, policy=DuplicatePolicy.SKIP)

        log.info("Documents indexed in InMemoryDocumentStore (duplicates skipped)")

        retriever = InMemoryBM25Retriever(document_store=document_store, top_k=TOP_K)

        log.info("BM25 retriever initialized")

        template = """
        Given the following information, answer the question.

        Context:
        {% for document in documents -%}
          {{ document.content }}
        {% endfor %}
        Question: {{question}}
        Answer:
        """

        prompt_builder = PromptBuilder(
            template=template, required_variables=["documents", "question"]
        )

        log.info("PromptBuilder initialized")

        ollama_generator = OllamaGenerator(
            model=OLLAMA_MODEL,
            url=OLLAMA_SERVER_URL,
            timeout=120,
            generation_kwargs={
                "num_predict": 256,
                "temperature": 0.2,
                "num_ctx": 1024,
                "top_p": 0.9,
            },
        )

        log.info(f"OllamaGenerator initialized | (model={OLLAMA_MODEL} | timeout=120s)")

        self.pipeline = Pipeline()

        self.pipeline.add_component("retriever", retriever)
        self.pipeline.add_component("prompt_builder", prompt_builder)
        self.pipeline.add_component("llm", ollama_generator)

        self.pipeline.connect("retriever", "prompt_builder.documents")
        self.pipeline.connect("prompt_builder", "llm")

        elapsed = time.time() - start_time
        log.info(f"Pipeline setup completed in {elapsed:.2f}s")

    def run_api(self, **kwargs: Any) -> dict[str, Any]:
        question: str = kwargs.get("question", "").strip()

        log.trace(f"run_api called with kwargs={kwargs}")

        if not question:
            log.error("run_api called without a question")
            raise ValueError("The 'question' field is required.")

        top_k: int = int(kwargs.get("top_k", TOP_K))
        log.debug(f"run_api using top_k={top_k}")

        start = time.time()
        result = self.pipeline.run(
            {
                "retriever": {"query": question, "top_k": top_k},
                "prompt_builder": {"question": question},
            }
        )

        reply = result["llm"]["replies"][0]
        elapsed = time.time() - start

        log.info(f"run_api completed in {elapsed:.2f}s | question='{question[:80]}'")

        return {"reply": reply}

    def run_chat_completion(
        self,
        model: str,
        messages: list[dict],
        body: dict,
    ) -> Union[str, Generator]:
        log.trace(f"run_chat_completion called | model={model} | body={body}")

        question = ""
        for msg in reversed(messages):
            if msg.get("role") == "user" and msg.get("content"):
                question = str(msg["content"]).strip()
                break

        if not question:
            log.warning("No user message found in chat history")
            return "No question provided."

        top_k = int(body.get("top_k", TOP_K))

        log.info(f"Streaming RAG run | question='{question[:80]}' | top_k={top_k}")

        return streaming_generator(
            pipeline=self.pipeline,
            pipeline_run_args={
                "retriever": {"query": question, "top_k": top_k},
                "prompt_builder": {"question": question},
            },
        )
