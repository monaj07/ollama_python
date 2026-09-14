# Ollama Python

A collection of notebooks for exploring local language models with the Ollama Python SDK.

The [Week 1 notebook](src/week_01.ipynb) covers:

- Sending chat requests and inspecting model thinking.
- Creating a reusable LLM backend interface.
- Building an interactive conversation loop with message history.
- Measuring response time, token counts, and generation speed.

The [Week 2 notebook](src/week_02.ipynb) builds on Week 1 with:

- Defining addition and division functions as tools for the model.
- Guiding tool selection with descriptions and a general system instruction.
- Executing requested tools and preserving their requests and results in message history.
- Handling multiple tool rounds, with a five-round limit, before returning a text answer.

Try `What is 10 / (3 + 2)?` to explore a calculation that can use both tools.

## Getting started

Use Python 3.14 or newer with the `ollama` package installed and a Jupyter-compatible editor, such as VS Code. Start Ollama locally and download the models used in the notebooks (`qwen3.5:2b` and `gemma4:e2b`), or update the model names to ones installed.

Start with `src/week_01.ipynb`, then explore tool calling in `src/week_02.ipynb`. Run each notebook's cells in order. Enter `exit` or `quit` to end a conversation loop.
