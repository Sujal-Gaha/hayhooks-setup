FROM deepset/hayhooks:main

# Install required dependencies for pipeline
RUN python -m pip install --no-cache-dir \
    pypdf \
    pypdf2 \
    nltk \
    haystack-ai \
    ollama-haystack \
    python-dotenv \
    trafilatura \
    chromadb \
    chroma-haystack

# Directories for ChromaDB persistence
RUN mkdir -p /data/chroma_db

# No need for start.sh anymore, use default command
CMD ["hayhooks", "run", "--host", "0.0.0.0"]
