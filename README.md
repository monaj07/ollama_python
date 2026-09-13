# Ollama Python

A collection of notebooks for exploring local language models with the Ollama Python SDK.

The [Week 1 notebook](src/week_01.ipynb) covers:

- Sending chat requests and inspecting model thinking.
- Creating a reusable LLM backend interface.
- Building an interactive conversation loop with message history.
- Measuring response time, token counts, and generation speed.

## Getting started

Use Python 3.14 or newer with the `ollama` package installed and a Jupyter-compatible editor, such as VS Code. Start Ollama locally and download the models used in the notebook (`qwen3.5:2b` and `gemma4:e2b`), or update the model names to ones installed.

Open `src/week_01.ipynb` and run the cells in order. Enter `exit` or `quit` to end a conversation loop.
