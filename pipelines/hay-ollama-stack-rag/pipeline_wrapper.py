from typing import Any, Dict, Generator, Union
from datasets import load_dataset
from hayhooks import BasePipelineWrapper, streaming_generator
from haystack import Document, Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy
from haystack.components.retrievers import InMemoryBM25Retriever
from haystack.components.builders import PromptBuilder
from haystack_integrations.components.generators.ollama import OllamaGenerator
from hayhooks.server.logger import log


class PipelineWrapper(BasePipelineWrapper):
    def setup(self):
        dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split="train[:5000]")

        docs = []

        for doc in dataset:
            titles = doc["context"]["title"]
            sentences_list = doc["context"]["sentences"]

            for title, sentences in zip(titles, sentences_list):
                content = f"{title}: {" ".join(sentences)}"

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

        document_store = InMemoryDocumentStore()
        document_store.write_documents(docs, policy=DuplicatePolicy.SKIP)

        retriever = InMemoryBM25Retriever(document_store, top_k=10)

        template = """
        Given the following information, answer the question.

        Context:
        {% for document in documents %}
            {{ document.content }}
        {% endfor %}
        Question: {{question}}
        Answer:
        """

        prompt_builder = PromptBuilder(
            template, required_variables=["documents", "question"]
        )

        ollama_generator = OllamaGenerator(
            model="llama3.1:8b",
            # model="gpt-oss:latest",
            url="http://ollama:11434",
            timeout=120,
            generation_kwargs={
                "num_predict": 1024,
                "temperature": 0.7,
                "num_ctx": 8192,
            },
            # streaming_callback=print_streaming_chunk,
        )

        self.pipeline = Pipeline()

        self.pipeline.add_component("retriever", retriever)
        self.pipeline.add_component("prompt_builder", prompt_builder)
        self.pipeline.add_component("llm", ollama_generator)

        self.pipeline.connect("retriever", "prompt_builder.documents")
        self.pipeline.connect("prompt_builder", "llm")

    def run_api(self, **kwargs: Any) -> Dict[str, Any]:
        """Define the inputs expected by Open WebUI chat interface"""

        question: str = kwargs.get("question", "").strip()

        if not question:
            raise ValueError("The 'question' field is required.")

        top_k: int = int(kwargs.get("top_k", 10))

        result = self.pipeline.run(
            {
                "retriever": {"query": question, "top_k": top_k},
                "prompt_builder": {"question": question},
            }
        )

        reply = result["llm"]["replies"][0]

        return {"reply": reply}

    def run_chat_completion(
        self,
        model: str,
        messages: list[dict],
        body: dict,
    ) -> Union[str, Generator]:
        question = ""
        for msg in reversed(messages):
            if msg.get("role") == "user" and msg.get("content"):
                question = str(msg["content"]).strip()
                break

        if not question:
            return "No question provided."

        top_k = int(body.get("top_k", 10))

        log.trace(f"Running RAG pipeline for question: {question}")

        return streaming_generator(
            pipeline=self.pipeline,
            pipeline_run_args={
                "retriever": {"query": question, "top_k": top_k},
                "prompt_builder": {"question": question},
            },
        )
