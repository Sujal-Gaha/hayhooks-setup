FROM deepset/hayhooks:main

# Install required dependencies pipeline
RUN pip install --no-cache-dir \
    pypdf \
    pypdf2 \
    nltk \
    datasets \
    haystack-ai \
    ollama-haystack \
    python-dotenv \
    trafilatura \
    chromadb \
    chroma-haystack \
    sentence-transformers \
    torch

# Directories for ChromaDB persistence
RUN mkdir -p /data/chroma_db

CMD ["hayhooks", "run", "--host", "0.0.0.0"]

