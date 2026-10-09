# WhatsApp Imitation Agent

This is a personal passion project: an impersonator chatbot built to help me learn about vector RAG, agentic AI, and prompt engineering. It uses a WhatsApp chat export as context and attempts to reply in the style of a selected participant. It does not fine-tune a model.

The current parser handles WhatsApp text exports in English or Dutch, including private and group chats that use the supported timestamp format. The repository includes a synthetic English group chat with three participants so you can try the pipeline without using a personal export.

## How it works

1. The parser reads WhatsApp messages and skips system notices. It converts omitted attachment markers, such as “image omitted” and “afbeelding weggelaten,” into placeholders.
2. Messages are sorted and grouped into sessions. A gap longer than `SESSION_GAP_MINUTES` starts a new session. Each session becomes a document containing the messages and metadata.
3. The index builder writes the session documents to a local Chroma database for semantic search. It also saves the documents as `chroma_db/bm25_documents.json`; the BM25 retriever uses that file for lexical keyword search. Both retrieval methods search sessions, and both are run for each agent response.
4. The LangGraph workflow plans a semantic query and keywords, retrieves relevant sessions through Chroma and BM25, drafts a reply, and retrieves random sessions as style examples. A review step can request another random sample if it thinks one may help. `MAX_RANDOM_CHECKS` sets the loop limit. The agent then returns its final message.

The chat history is kept in memory for the current running process and conversation thread. It is not added to the index, saved as long-term memory, or retained after the process ends.

## Setup

Create and activate the Conda environment, then make your local `.env` file:

```bash
conda env create -f environment.yml
conda activate imitation_agent
cp .env.example .env
```

Add your OpenAI API key to `.env`. The example configuration uses Taylor as the impersonated participant. Change `IMPERSONATED_NAME` to the sender name used in your chat. The agent loads your local `src/prompts.py` if present; otherwise, it uses the public `src/prompts_example.py`.

## Add a chat and build the index

Put a WhatsApp `.txt` export in `data/`, then build the Chroma database and BM25 JSON file:

```bash
python -m scripts.build_index data/my_chat.txt
```

The included sample can be indexed with:

```bash
python -m scripts.build_index data/example_chat.txt
```

Indexing calls the configured OpenAI embedding model and writes the database and BM25 corpus under `chroma_db/`. The database is local and ignored by Git. Only process chats you have permission to use; message text is sent to the configured embedding API during indexing and may be sent to the chat model as retrieved context during agent use.

## Run the agent

After building the index, start the chat loop:

```bash
python -m src.agent
```

Type `exit` to stop. The agent needs a valid API key and a built index.

## Inspect, test, and evaluate

Inspect a chat export without printing message contents:

```bash
python -m scripts.inspect_data data/my_chat.txt
```

Run the parser and session tests:

```bash
python -m unittest discover -s tests
```

Run the evaluation cases:

```bash
python -m src.evaluation.run_eval
```

This runs the agent on fixed questions and reports keyword coverage. It uses your local `evals/questions.jsonl` if present, or the public `evals/questions.example.jsonl` otherwise. These evaluations are a separate manual check; their scores are not part of the live agent workflow. Generated answers are written under the ignored `evals/results/` directory.

## Notebook workflow

Open `notebooks/workflow_playground.ipynb` and run the cells in order. The notebook selects a local `.txt` export under `data/` if one is present, otherwise it uses the included example chat. Indexing, model calls, and tests are off by default; enable the relevant flags in section 0. Build an index before running retrieval or the agent. The notebook is useful for inspecting parsed data, checking retrieval results, trying agent responses, and experimenting with prompts.

## Configuration

Put local settings in `.env`; `.env.example` lists the same variables with public defaults. `src/config.py` loads them for the Python code.

| Variable | What it controls |
| --- | --- |
| `OPENAI_API_KEY` | API credential used by the chat and embedding models. |
| `IMPERSONATED_NAME` | Target sender name, used in prompts and the terminal label. It should match the name in the export. |
| `MODEL_NAME` | Chat model used for query planning, drafting, and reviewing. |
| `EMBEDDING_MODEL` | Embedding model used to build and query Chroma. |
| `TEMPERATURE` | Chat model response randomness. |
| `MAX_TOKENS` | Maximum completion tokens for a model response. |
| `SESSION_GAP_MINUTES` | Message time gap that separates sessions. |
| `VECTOR_SEARCH_K` | Number of sessions returned by semantic search. |
| `BM25_SEARCH_K` | Number of sessions returned by BM25 lexical search. |
| `MAX_RANDOM_CHECKS` | Maximum number of times the review loop can request another random style sample. |
| `RANDOM_SESSIONS_PER_CHECK` | Number of random sessions retrieved each time. |
| `CHUNK_SIZE`, `CHUNK_OVERLAP` | Reserved settings; currently unused because indexing stores one document per session rather than splitting sessions into chunks. |

## Prompt engineering

The main system prompt is `SYSTEM_PROMPT` in `src/prompts.py`. That file is ignored by Git so you can keep a private local prompt. For a public clone, start from `src/prompts_example.py`; copy it to `src/prompts.py` for a local prompt, or edit the example prompt directly if you intend to publish your changes.

There are also task-specific prompts in `src/agent.py`: the semantic query planner, BM25 keyword extraction, answer drafting instructions, and answer review instructions. Tool descriptions in `src/tools.py` guide how the model uses retrieval tools. These are useful places to experiment alongside the system prompt. The notebook supports manually trying agent responses and reviewing retrieval output.

## Future work

Possible next steps include support for more chat formats, such as Discord exports; additional retrieval tools; persistent memory across runs; and performance improvements. This version is a prototype focused on getting the parsing, retrieval, and agent workflow working end to end.

## Privacy

`.env`, `src/prompts.py`, personal chat exports under `data/`, the Chroma/BM25 index, private evaluation questions, and evaluation results are ignored by Git. The repository includes only the synthetic chat and example prompt/evaluation cases. Check `git status` before committing so you can confirm that personal files are not staged.
