# Hayhooks with Open WebUI - Docker Compose Setup

This repository contains a Docker Compose configuration to run [Haystack](https://www.haystack.com) pipelines with [Hayhooks](https://github.com/deepset-ai/hayhooks) using also [Open WebUI](https://github.com/open-webui/open-webui) for a complete chat interface experience.

In `pipelines` folder there's a ready-to-use wrapper for [chat_with_website](https://docs.haystack.deepset.ai/docs/pipeline-templates#chat-with-website) pipeline.

That folder will be mounted in the `/pipelines` directory of Hayhooks service.

## Requirements

- [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/) installed on your system
- Git (to clone this repository)
- An `OPENAI_API_KEY` environment variable set (needed for the `chat_with_website` pipeline)

## Run

Clone the repository:

```bash
git clone https://github.com/deepset-ai/hayhooks-open-webui-docker-compose.git
```

Run the docker compose file:

```bash
docker compose up --build
```

And that's it! 😉

You can now access to Hayhooks on `http://localhost:1416` and to Open WebUI on `http://localhost:3000`. The `chat_with_website_streaming` pipeline will be available on Open WebUI.

Note that `open-webui` _may require some time to start up_.

## Tear down

To tear down the environment, run:

```bash
docker compose down
```

## Additional notes

### `Dockerfile`

- The `Dockerfile` is based on the `deepset/hayhooks:main` image.
- It installs `trafilatura` as a dependency, needed by the `chat_with_website` pipeline.

### `docker-compose.yml`

- The `docker-compose.yml` file mounts the `pipelines` folder in the `/pipelines` directory of Hayhooks service.
- The `OPENAI_API_KEY` environment variable is passed to Hayhooks service.

About `open-webui` settings:

- The `OPENAI_API_BASE_URL` environment variable is set to `http://hayhooks:1416`, pointing to Hayhooks service.
- The `OPENAI_API_KEY` environment variable is set to a dummy value (`"dummy"`), as it's not needed for Hayhooks service.
- The `WEBUI_AUTH` environment variable is set to `false`, as we don't need authentication for this demo.
- The `ENABLE_TAGS_GENERATION` and `ENABLE_EVALUATION_ARENA_MODELS` environment variables are set to `false`, as they are not needed for Hayhooks service.
