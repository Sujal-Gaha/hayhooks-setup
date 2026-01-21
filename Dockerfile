FROM deepset/hayhooks:main

# Install required dependencies pipeline
RUN pip install --no-cache-dir \
    datasets \
    haystack-ai \
    ollama-haystack \
    python-dotenv \
    chromadb \
    chroma-haystack \
    sentence-transformers \
    trafilatura

# Create directories for ChromaDB persistence
RUN mkdir -p /data/chroma_db

CMD ["hayhooks", "run", "--host", "0.0.0.0"]

