"""Retrieval, answer generation, and the Gradio interface for CampusRAG."""

from __future__ import annotations

import html
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import gradio as gr
from openai import OpenAI

from .vector_store import (
    COLLECTION_NAME,
    DEFAULT_VECTOR_DB_PATH,
    dense_search,
    load_embedding_model,
    open_qdrant,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEHRAN_TIMEZONE = ZoneInfo("Asia/Tehran")
GENERATION_MODEL = "gemma4:e2b"
OLLAMA_BASE_URL = "http://127.0.0.1:11434/v1"


def format_date(value: Any) -> str:
    """Format an ISO date in Tehran time and show its approximate age."""
    if not value:
        return "Unknown date"

    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)

        local_time = parsed.astimezone(TEHRAN_TIMEZONE)
        age_days = max(
            0,
            (datetime.now(timezone.utc) - parsed.astimezone(timezone.utc)).days,
        )

        if age_days < 30:
            age = f"{age_days} days ago"
        elif age_days < 365:
            age = f"{age_days // 30} months ago"
        else:
            age = f"{age_days // 365} years ago"

        return f"{local_time:%Y-%m-%d %H:%M} ({age})"
    except (TypeError, ValueError):
        return str(value)


def _parse_datetime(value: Any) -> datetime | None:
    if not value:
        return None

    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


def build_telegram_link(
    source_id: Any,
    message_id: Any,
    source_username: Any = None,
) -> str | None:
    """Build a Telegram link when the source supports one."""
    if message_id is None:
        return None

    username = str(source_username or "").strip().lstrip("@")
    if username:
        return f"https://t.me/{username}/{message_id}"

    source_id = str(source_id or "")
    if source_id.startswith("-100"):
        return f"https://t.me/c/{source_id[4:]}/{message_id}"

    return None


def _message_sources(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    """Return per-message metadata and support older vector records."""
    sources = metadata.get("message_sources")
    if isinstance(sources, list) and sources:
        return sources

    message_ids = metadata.get("message_ids") or []
    media_paths = metadata.get("media_paths") or []
    message_date = metadata.get("message_date")

    return [
        {
            "message_id": message_id,
            "message_date": message_date,
            "media_path": media_paths[index]
            if index < len(media_paths)
            else None,
        }
        for index, message_id in enumerate(message_ids)
    ]


def result_sources(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Add dates, Telegram links, and media paths to one result."""
    metadata = result.get("metadata") or {}
    source_id = metadata.get("source_id")
    source_username = metadata.get("source_username")

    return [
        {
            "message_id": message.get("message_id"),
            "date": format_date(message.get("message_date")),
            "message_date": message.get("message_date"),
            "link": build_telegram_link(
                source_id,
                message.get("message_id"),
                source_username,
            ),
            "media_path": message.get("media_path"),
        }
        for message in _message_sources(metadata)
    ]


def _latest_result_date(result: dict[str, Any]) -> datetime | None:
    dates = [
        _parse_datetime(source.get("message_date"))
        for source in result_sources(result)
    ]
    dates = [date for date in dates if date is not None]
    return max(dates) if dates else None


def _freshness_score(
    result: dict[str, Any],
    half_life_days: int = 180,
) -> float:
    latest = _latest_result_date(result)
    if latest is None:
        return 0.0

    age_days = max(
        0.0,
        (datetime.now(timezone.utc) - latest).total_seconds() / 86400,
    )
    return 2 ** (-age_days / half_life_days)


def rerank_with_freshness(
    results: list[dict[str, Any]],
    freshness_weight: float = 0.20,
    half_life_days: int = 180,
) -> list[dict[str, Any]]:
    """Give recent messages a small bonus after dense retrieval."""
    ranked = []

    for result in results:
        semantic_score = float(result.get("score", 0.0))
        semantic_score = (semantic_score + 1.0) / 2.0
        freshness = _freshness_score(result, half_life_days)
        combined_score = (
            (1 - freshness_weight) * semantic_score
            + freshness_weight * freshness
        )
        ranked.append((combined_score, result))

    ranked.sort(key=lambda item: item[0], reverse=True)
    return [result for _, result in ranked]


def build_context(results: list[dict[str, Any]]) -> str:
    """Build the small, dated context sent to the language model."""
    if not results:
        return "No relevant sources were found."

    context_parts = []

    for index, result in enumerate(results, start=1):
        metadata = result.get("metadata") or {}
        source_name = metadata.get("source_name", "Unknown source")
        source_details = result_sources(result)
        dates = ", ".join(item["date"] for item in source_details)
        links = ", ".join(
            item["link"]
            for item in source_details
            if item["link"]
        )

        context_parts.append(
            f"[Source {index}]\n"
            f"Source: {source_name}\n"
            f"Date(s): {dates or 'Unknown date'}\n"
            f"Telegram link(s): {links or 'Unavailable'}\n"
            f"Text:\n{result.get('text', '')}"
        )

    return "\n\n".join(context_parts)


def build_sources_markdown(results: list[dict[str, Any]]) -> str:
    """Format sources for the right side of the chat interface."""
    if not results:
        return "### Retrieved sources\n\nNo sources found."

    parts = ["### Retrieved sources"]

    for index, result in enumerate(results, start=1):
        metadata = result.get("metadata") or {}
        source_name = html.escape(
            str(metadata.get("source_name", "Unknown source"))
        )
        score = float(result.get("score", 0.0))
        parts.append(f"**Source {index} — {source_name} — score: {score:.4f}**")

        for source in result_sources(result):
            details = f"- Date: {source['date']}"
            if source["link"]:
                details += f" — [Open Telegram message]({source['link']})"
            parts.append(details)

        text = str(result.get("text", "")).strip()
        parts.append(f"> {text.replace(chr(10), chr(10) + '> ')}")

    return "\n\n".join(parts)


def retrieved_images(results: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """Return existing media files for the right-side gallery."""
    images = []
    seen = set()

    for index, result in enumerate(results, start=1):
        for source in result_sources(result):
            media_path = source.get("media_path")
            if not media_path:
                continue

            raw_path = Path(media_path)
            candidates = [raw_path]
            if not raw_path.is_absolute():
                candidates.append(PROJECT_ROOT / raw_path)

            path = next(
                (candidate for candidate in candidates if candidate.exists()),
                candidates[-1],
            ).resolve()

            if not path.exists() or path in seen:
                continue

            seen.add(path)
            images.append((str(path), f"Source {index} — {source['date']}"))

    return images


SYSTEM_PROMPT = """
You are a helpful university information assistant.
Answer in Persian.
Use only the provided sources and conversation history.
If the sources are not enough, say that you do not have enough information.
Do not invent names, dates, courses, links, or opinions.
""".strip()


def build_prompt(
    query: str,
    context: str,
    conversation_history: str = "",
) -> str:
    """Build the grounded prompt sent to Ollama."""
    return f"""
{SYSTEM_PROMPT}

Context:
{context}

Conversation history:
{conversation_history or 'No previous conversation.'}

Current user question:
{query}

Answer in Persian:
""".strip()


def _format_chat_history(
    history: list[dict[str, Any]] | None,
    max_messages: int = 6,
) -> str:
    messages = []

    for message in (history or [])[-max_messages:]:
        if not isinstance(message, dict):
            continue

        role = message.get("role")
        content = str(message.get("content", "")).strip()
        if role in {"user", "assistant"} and content:
            messages.append(f"{role}: {content}")

    return "\n".join(messages)


def _build_retrieval_query(
    query: str,
    history: list[dict[str, Any]] | None,
) -> str:
    previous_questions = [
        str(message.get("content", "")).strip()
        for message in (history or [])
        if isinstance(message, dict)
        and message.get("role") == "user"
        and message.get("content")
    ][-2:]

    return "\n".join(previous_questions + [query])


def create_gradio_app(
    *,
    model_name: str = "Alibaba-NLP/gte-multilingual-base",
    generation_model: str = GENERATION_MODEL,
    ollama_base_url: str = OLLAMA_BASE_URL,
    vector_db_path: Path = DEFAULT_VECTOR_DB_PATH,
    collection_name: str = COLLECTION_NAME,
):
    """Create the local CampusRAG chat application."""
    embedding_model = load_embedding_model(model_name=model_name)
    client = open_qdrant(vector_db_path, collection_name=collection_name)
    ollama = OpenAI(api_key="ollama", base_url=ollama_base_url)

    def answer_query(query: str, history: list[dict[str, Any]] | None):
        history = history or []
        query = (query or "").strip()

        if not query:
            return history, [], "Please enter a question.", query

        retrieval_query = _build_retrieval_query(query, history)
        results = dense_search(
            query=retrieval_query,
            embedding_model=embedding_model,
            client=client,
            collection_name=collection_name,
            top_k=50,
        )
        results = rerank_with_freshness(results)[:10]

        context = build_context(results)
        prompt = build_prompt(
            query,
            context,
            _format_chat_history(history),
        )
        sources = build_sources_markdown(results)

        try:
            completion = ollama.chat.completions.create(
                model=generation_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            answer = completion.choices[0].message.content
        except Exception as error:
            answer = (
                f"Generation failed: {error}\n\n"
                f"Make sure Ollama is running and {generation_model} is installed."
            )

        answer = answer or "The model returned an empty answer."
        history = history + [
            {"role": "user", "content": query},
            {"role": "assistant", "content": answer},
        ]

        return history, retrieved_images(results), sources, ""

    with gr.Blocks(title="CampusRAG") as app:
        gr.Markdown("# CampusRAG\nAsk questions about the university data.")

        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(type="messages", height=620)
                with gr.Row():
                    query_box = gr.Textbox(
                        label="Question",
                        placeholder="Ask about a professor, course, or announcement...",
                        scale=5,
                    )
                    send_button = gr.Button("Send", variant="primary", scale=1)
                gr.ClearButton(
                    [query_box, chatbot],
                    value="Clear",
                )

            with gr.Column(scale=1):
                gallery = gr.Gallery(
                    label="Images from retrieved messages",
                    columns=1,
                    height=620,
                    object_fit="contain",
                )
                sources_box = gr.Markdown(label="Sources")

        outputs = [chatbot, gallery, sources_box, query_box]
        send_button.click(
            answer_query,
            inputs=[query_box, chatbot],
            outputs=outputs,
        )
        query_box.submit(
            answer_query,
            inputs=[query_box, chatbot],
            outputs=outputs,
        )

    return app
