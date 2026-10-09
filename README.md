# WhatsApp Imitation Agent

A small conversational agent that uses a WhatsApp chat export as context for answering in a selected participant's style. It indexes chat sessions in Chroma for semantic retrieval and saves a BM25 corpus for lexical retrieval. LangGraph combines both retrieval paths with a drafting and review step.

The repository includes a made-up English group chat and a public example prompt for **Taylor**. The example files let you try the workflow without a personal chat export. Your own chat, API key, and private prompt should stay local.

## Setup

Create and activate the Conda environment, then make a local environment file:

```bash
conda env create -f environment.yml
conda activate imitation_agent
cp .env.example .env
```

Edit `.env` and set `OPENAI_API_KEY`. You can also change `IMPERSONATED_NAME` and the model and chunk settings there. Do not commit `.env`.

## Build the index

Build the default local Chroma database and BM25 corpus from the included sample chat:

```bash
python -m scripts.build_index data/example_chat.txt
```

For a personal export, pass its path instead. WhatsApp exports are expected to be plain text in the format handled by `src/data/whatsapp_parser.py`. Keep personal exports under `data/`; they are ignored by Git except for `data/example_chat.txt`.

Indexing sends chat text to the configured OpenAI embedding API. Only index conversations you are allowed to process and share with that service. The index is written to the local, Git-ignored `chroma_db/` directory.

## Run the agent

After building an index:

```bash
python -m src.agent
```

Enter a message at the prompt. Type `exit` to stop. The agent needs the API key and a built local index. It uses `src/prompts_example.py` when the private `src/prompts.py` is absent, as it will be for a GitHub clone.

To use a private prompt locally, create `src/prompts.py`; that path is ignored by Git. Set its `SYSTEM_PROMPT` value to your prompt. The target name is read from `IMPERSONATED_NAME` in `.env`.

## Inspect, evaluate, and test

Inspect an export without printing message contents:

```bash
python -m scripts.inspect_data data/example_chat.txt
```

Run the example evaluation cases:

```bash
python -m src.evaluation.run_eval
```

The command uses `evals/questions.jsonl` if you have a local private set; otherwise it falls back to the public `evals/questions.example.jsonl`. Evaluation calls the chat model and writes generated answers under the ignored `evals/results/` directory.

Run the parser and session unit tests:

```bash
python -m unittest discover -s tests
```

## Notebook

Open `notebooks/workflow_playground.ipynb` from this repository. It selects a local text export under `data/` if one is present; otherwise it uses the example chat. Index building, model calls, and tests are off by default. Enable the steps you want in notebook section 0, and build an index before running retrieval or the agent.

## What is kept local

Git ignores `.env`, `src/prompts.py`, personal chat and JSON exports under `data/`, the Chroma/BM25 index, private evaluation cases, and evaluation results. The tracked `.env.example`, `src/prompts_example.py`, `data/example_chat.txt`, and `evals/questions.example.jsonl` are safe starting points for a clone. Check `git status` before committing to confirm that no personal files are staged.
