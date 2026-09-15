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

The [Week 2 extended notebook](src/week_02_extended.ipynb) adds:

- A `save_to_file` tool for writing model-provided content to a local file.
- Pydantic input models for addition, division, and file saving.
- Validation of tool arguments before execution, including rejecting division by zero.
- Passing validated arguments to each tool and returning validation errors to the model as tool results.

Try `Save the text "Hello from Ollama" to answer.log` to explore file saving. The file is written relative to the notebook kernel's working directory; an existing file with the same name is overwritten.

The [Week 3 notebook](src/week_03.ipynb) adds a step-by-step SQLite memory layer:

- Sessions and saved user/assistant messages.
- Persistent facts with a simple parameterized `LIKE` search.
- Tools to remember facts, search memories, create tasks, and list tasks.
- An in-memory example you can run without Ollama, followed by the interactive chat.

Run its numbered steps in order. The persistent database is `week_03_memory.sqlite3`
in the kernel's working directory (the notebook prints its full path). Reuse that
file across runs to retain memories and tasks. Each chat starts a new session;
stored chat history is available for inspection but is not automatically reloaded
into the model. Local database files are ignored by Git.

## Getting started

Use Python 3.14 or newer with the `ollama` package installed and a Jupyter-compatible editor, such as VS Code. The extended and Week 3 notebooks also require Pydantic 2. Start Ollama locally and download the models used in the notebooks (`qwen3.5:2b` and `gemma4:e2b`), or update the model names to ones installed.

Start with `src/week_01.ipynb`, then explore tool calling in `src/week_02.ipynb` and input validation in `src/week_02_extended.ipynb`. Run each notebook's cells in order. Enter `exit` or `quit` to end a conversation loop.

## Troubleshooting

`ValueError: No content returned from the model.` is raised by the notebook when an Ollama response contains neither text nor tool calls. It can occur on the follow-up response after a tool has already completed, including saving a file. Inspect the raw `output` and the `messages` inside `OllamaBackend.generate()` before the exception to investigate; this error alone does not identify why the response was empty.
