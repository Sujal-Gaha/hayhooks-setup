FROM deepset/hayhooks:main

# Install required dependencies for your custom pipeline + extras
RUN pip install --no-cache-dir \
    datasets \
    haystack-ai \
    ollama_haystack \
    trafilatura

# Default command (you can override in compose if needed)
CMD ["hayhooks", "run", "--host", "0.0.0.0"]
