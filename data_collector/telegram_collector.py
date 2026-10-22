import os
import re
import asyncio

from datetime import timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv
from telethon import TelegramClient

from .database import (
    open_database,
    get_last_message_id,
    update_source_username,
    save_message,
)


ROOT = Path(__file__).resolve().parent.parent

load_dotenv(ROOT / ".env")

API_ID = int(os.environ["TELEGRAM_API_ID"])
API_HASH = os.environ["TELEGRAM_API_HASH"]

SESSION_PATH = ROOT / "data" / "telegram_session"

MEDIA_PATH = ROOT / "data" / "media"

IRAN_TIMEZONE = timezone(
    timedelta(hours=3, minutes=30)
)


client = TelegramClient(
    str(SESSION_PATH),
    API_ID,
    API_HASH,
)


def safe_name(name):
    name = re.sub(r'[<>:"/\\|?*]', "_", name)
    return name.strip().rstrip(". ")


def is_image(message):
    if message.photo:
        return True

    if message.document:
        mime_type = message.document.mime_type

        if mime_type and mime_type.startswith("image/"):
            return True

    return False


def image_extension(message):
    if message.photo:
        return ".jpg"

    if message.file and message.file.ext:
        return message.file.ext

    return ".jpg"


def file_date(message):
    date = message.date.astimezone(IRAN_TIMEZONE)

    return date.strftime(
        "%Y-%m-%d_%H-%M-%S"
    )


async def save_channel_image(message, channel_name):
    if not is_image(message):
        return None, None

    folder = (
        MEDIA_PATH
        / safe_name(channel_name)
        / "images"
    )

    folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    base_name = (
        f"{message.id}_{file_date(message)}"
    )

    image_path = (
        folder
        / f"{base_name}{image_extension(message)}"
    )

    if not image_path.exists():
        await message.download_media(
            file=str(image_path)
        )

    caption_path = None

    if message.text:
        caption_path = folder / f"{base_name}.txt"

        caption_path.write_text(
            message.text,
            encoding="utf-8",
        )

    return str(image_path), (
        str(caption_path)
        if caption_path
        else None
    )


async def sync_source(db, dialog, source_type):
    source_id = str(dialog.id)
    source_name = dialog.name
    source_username = getattr(dialog.entity, "username", None)

    await update_source_username(
        db,
        "telegram",
        source_id,
        source_username,
    )

    last_id = await get_last_message_id(
    db,
    "telegram",
    source_id,
    )

    print()
    print(f"Syncing: {source_name}")

    if last_id:
        print(
            f"Continuing after message: {last_id}"
        )
    else:
        print("Full history sync")

    count = 0

    async for message in client.iter_messages(
        dialog.entity,
        min_id=last_id,
        reverse=True,
    ):
        media_path = None
        caption_path = None

        # Images are downloaded only from channels.
        if (
            source_type == "channel"
            and is_image(message)
        ):
            media_path, caption_path = (
                await save_channel_image(
                    message,
                    source_name,
                )
            )

        reply_id = None

        if message.reply_to:
            reply_id = (
                message.reply_to.reply_to_msg_id
            )

        await save_message(
            db,
            {
                "platform": "telegram",
                "source_type": source_type,
                "source_id": source_id,
                "source_name": source_name,
                "source_username": source_username,
                "message_id": message.id,
                "text": message.text,
                "message_date": (
                    message.date.isoformat()
                    if message.date
                    else None
                ),
                "edit_date": (
                    message.edit_date.isoformat()
                    if message.edit_date
                    else None
                ),
                "reply_to_message_id": reply_id,
                "media_path": media_path,
                "caption_path": caption_path,
            },
        )

        count += 1

        if count % 500 == 0:
            await db.commit()
            print(f"Saved: {count}")

    await db.commit()

    print(
        f"Finished: {source_name} | "
        f"{count} new messages"
    )


async def choose_sources(items, title):
    print()
    print(title)
    print("-" * 60)

    for index, dialog in enumerate(items, start=1):
        print(f"{index}. {dialog.name}")

    print("0. Skip")
    print("all. Select all")

    while True:
        value = input(
            "Select (example: 1,2,4 or all): "
        ).strip().lower()

        if value == "0":
            return []

        if value == "all":
            return items

        try:
            numbers = [
                int(x.strip())
                for x in value.split(",")
            ]

            if all(
                1 <= number <= len(items)
                for number in numbers
            ):
                return [
                    items[number - 1]
                    for number in numbers
                ]

        except ValueError:
            pass

        print("Invalid selection.")

async def main():
    MEDIA_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    db = await open_database()

    try:
        await client.start()

        channels = []
        groups = []

        async for dialog in client.iter_dialogs():
            if dialog.is_group:
                groups.append(dialog)

            elif dialog.is_channel:
                channels.append(dialog)

        selected_channels = await choose_sources(
            channels,
            "CHANNELS",
        )

        selected_groups = await choose_sources(
            groups,
            "GROUPS",
        )

        for channel in selected_channels:
            await sync_source(
                db,
                channel,
                "channel",
            )

        for group in selected_groups:
            await sync_source(
                db,
                group,
                "group",
            )

    finally:
        await db.close()


if __name__ == "__main__":
    with client:
        client.loop.run_until_complete(main())
