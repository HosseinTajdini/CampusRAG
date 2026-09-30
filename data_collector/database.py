from pathlib import Path

import aiosqlite


ROOT = Path(__file__).resolve().parent.parent

DB_PATH = (
    ROOT
    / "data"
    / "database"
    / "collected.db"
)


async def open_database():
    DB_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    db = await aiosqlite.connect(
        DB_PATH
    )

    db.row_factory = aiosqlite.Row

    await db.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,

            platform TEXT NOT NULL,
            source_type TEXT NOT NULL,
            source_id TEXT NOT NULL,
            source_name TEXT NOT NULL,

            message_id INTEGER NOT NULL,

            text TEXT,

            message_date TEXT,
            edit_date TEXT,

            reply_to_message_id INTEGER,

            media_path TEXT,
            caption_path TEXT,

            UNIQUE(
                platform,
                source_id,
                message_id
            )
        )
    """)

    await db.execute("""
        CREATE INDEX IF NOT EXISTS idx_messages_source
        ON messages(
            platform,
            source_id,
            message_id
        )
    """)

    await db.execute("""
        CREATE INDEX IF NOT EXISTS idx_messages_reply
        ON messages(
            platform,
            source_id,
            reply_to_message_id
        )
    """)

    await db.commit()

    return db


async def get_last_message_id(
    db,
    platform,
    source_id,
):
    cursor = await db.execute(
        """
        SELECT MAX(message_id)
        FROM messages
        WHERE platform = ?
          AND source_id = ?
        """,
        (
            platform,
            source_id,
        ),
    )

    row = await cursor.fetchone()

    if row and row[0]:
        return row[0]

    return 0


async def save_message(
    db,
    message,
):
    await db.execute(
        """
        INSERT INTO messages (
            platform,
            source_type,
            source_id,
            source_name,

            message_id,

            text,

            message_date,
            edit_date,

            reply_to_message_id,

            media_path,
            caption_path
        )
        VALUES (
            ?, ?, ?, ?,
            ?,
            ?,
            ?, ?,
            ?,
            ?, ?
        )

        ON CONFLICT(
            platform,
            source_id,
            message_id
        )
        DO UPDATE SET

            source_type =
                excluded.source_type,

            source_name =
                excluded.source_name,

            text =
                excluded.text,

            message_date =
                excluded.message_date,

            edit_date =
                excluded.edit_date,

            reply_to_message_id =
                excluded.reply_to_message_id,

            media_path =
                excluded.media_path,

            caption_path =
                excluded.caption_path
        """,
        (
            message["platform"],
            message["source_type"],
            message["source_id"],
            message["source_name"],

            message["message_id"],

            message["text"],

            message["message_date"],
            message["edit_date"],

            message["reply_to_message_id"],

            message["media_path"],
            message["caption_path"],
        ),
    )