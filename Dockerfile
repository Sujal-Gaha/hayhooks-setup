FROM deepset/hayhooks:main

# Install required dependencies pipeline
RUN pip install --no-cache-dir \
    datasets \
    haystack-ai \
    ollama-haystack \
    python-dotenv \
    trafilatura \
    chromadb \
    chroma-haystack \
    sentence-transformers \
    pypdf \
    pypdf2 \
    nltk \
    torch

# Create directories for ChromaDB persistence
RUN mkdir -p /data/chroma_db

CMD ["hayhooks", "run", "--host", "0.0.0.0"]

