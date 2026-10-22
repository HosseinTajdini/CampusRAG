# CampusRAG

CampusRAG is a small retrieval-augmented generation project for university information collected from Telegram.

It collects messages, cleans and groups them, creates multilingual embeddings, stores the vectors in local Qdrant, and answers questions with a local Ollama model.

## Features

- Telegram message collection with SQLite
- Text cleaning and reply-thread grouping
- Multilingual dense embeddings with `Alibaba-NLP/gte-multilingual-base`
- Local Qdrant vector database
- Freshness-aware result ranking
- Persian answers with date and source context
- Telegram message links and retrieved image previews
- Gradio chat interface with conversation history

## Pipeline

```text
Telegram
   -> SQLite
   -> cleaned documents
   -> embeddings
   -> Qdrant
   -> retrieval and freshness ranking
   -> Ollama
   -> Gradio chat
```

## Setup

Create a virtual environment and install the main dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install aiosqlite python-dotenv telethon pandas tqdm numpy qdrant-client sentence-transformers gradio openai
```

Create a `.env` file in the project root:

```env
TELEGRAM_API_ID=your_api_id
TELEGRAM_API_HASH=your_api_hash
```

Telegram sources are listed in `source.txt`. The collector currently lets you select channels and groups interactively.

Install and start Ollama, then download the generation model:

```powershell
ollama pull gemma4:e2b
```

## Run

Run the complete pipeline from the project root:

```powershell
python main.py
```

`main.py` performs these steps:

1. Collect new Telegram messages.
2. Build cleaned documents.
3. Create embeddings and update Qdrant.
4. Start the Gradio chat interface.

The first embedding-model load can take time. Later runs use the Hugging Face cache.

## Project structure

```text
CampusRAG/
├── main.py
├── source.txt
├── data_collector/
│   ├── database.py
│   └── telegram_collector.py
├── data_processing/
│   ├── build_documents.py
│   ├── rag_pipeline.py
│   └── vector_store.py
├── notebooks/
│   ├── 01_data_audit.ipynb
│   ├── 02_chunking.ipynb
│   └── 03_rag_pipeline.ipynb
└── data/
    ├── database/
    ├── media/
    ├── processed/
    └── vector_db/
```

## RAG behavior

The system retrieves 50 dense candidates, gives recent messages a small freshness bonus, and sends the best 10 results to `gemma4:e2b` as context. Conversation history is kept during the current Gradio session and is included in follow-up questions.

The right side of the interface contains the retrieved sources, dates, Telegram links, and available images. The chat area contains only the generated answer.

## Notes

- Local data, Telegram sessions, model files, and vector storage are ignored by Git.
- Never commit `.env` or Telegram session files.
- The notebooks explain the pipeline step by step before it is used from Python modules.
