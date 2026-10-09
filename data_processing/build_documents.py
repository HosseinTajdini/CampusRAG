import sqlite3
import pandas as pd 
import re

from tqdm import tqdm
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "database" / "collected.db"


def get_database(database_direction:Path):
    db = sqlite3.connect(database_direction)
    df = pd.read_sql_query(
    """
    SELECT * 
    FROM messages
    """,
    db
    )
    return df


#Cleaning Data
def cleaning_data(df:pd.DataFrame):
    df.drop(df[df['caption_path'].isna()&df['media_path'].notna()].index, inplace=True)#picture wait out caption usualy nonsenes 
    df.drop(df[(df['source_name']== "شمسی نیوز") & df['media_path'].notna()].index, inplace=True)# its full of meme
    df.drop(columns=['edit_date','caption_path'],inplace=True)# no need
    df = df[df["text"].notna() & (df["text"].str.strip() != "")].copy()#why not?
    df["content"] = df["text"].apply(normalize_text)# make it easier for vocab
    
    return df



def normalize_text(text):
    text = str(text)

    # Arabic characters -> Persian
    text = text.replace("ي", "ی").replace("ك", "ک")

    # Normalize spaces
    text = re.sub(r"[ \t]+", " ", text)

    # Too many blank lines -> max one blank line
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    return text.strip()


def split_data(df:pd.DataFrame):
    channels = df[df["source_type"] == "channel"].copy()
    groups = df[df["source_type"] == "group"].copy()

    groups["source_id"] = pd.to_numeric(
        groups["source_id"],
        errors="coerce" #if its empty just make it NaN
    ).astype("Int64")

    groups["message_id"] = pd.to_numeric(
        groups["message_id"],
        errors="coerce"
    ).astype("Int64")

    groups["reply_to_message_id"] = pd.to_numeric(
        groups["reply_to_message_id"],
        errors="coerce"
    ).astype("Int64")

    return channels,groups

def build_parent_map(groups:pd.DataFrame):
    parent_map = {}

    for row in tqdm(groups.itertuples(),
                total=len(groups),
                desc="Building parent map"):
        
        key = (int(row.source_id),int(row.message_id))

        if pd.notna(row.reply_to_message_id):
            value = int(row.reply_to_message_id)
        else:
            value = None

        parent_map[key] = value

    return parent_map

def find_root(parent_map,source_id,message_id):
    source_id = int(source_id)
    current = int(message_id)
    visited = set()

    while True:
        key = (source_id,current)

        if key in visited:
            break

        visited.add(key)

        parent = parent_map.get(key)

        if parent is None:
            break

        if (source_id,parent) not in parent_map:
            break

        current = parent

    return current

def add_root_message(groups:pd.DataFrame,parent_map):
    roots = []

    for source_id,message_id in tqdm(
        zip(groups["source_id"],groups["message_id"]),
        total=len(groups),
        desc="Finding thread roots"
    ):
        roots.append(
            find_root(parent_map,source_id,message_id)
        )

    groups["root_message_id"] = roots

    return groups

def build_channel_documents(channels:pd.DataFrame):
    channel_documents = channels[
        [
            "source_id",
            "source_name",
            "message_id",
            "message_date",
            "content",
            "media_path"
        ]
    ].copy()

    channel_documents["document_type"] = "channel_message"

    channel_documents["document_id"] = (
        channel_documents["source_id"].astype(str)
        + "_"
        + channel_documents["message_id"].astype(str)
    )

    return channel_documents

def build_thread_document(thread):
    thread = thread.sort_values("message_date")

    parts = []

    for row in thread.itertuples():
        parts.append(
            f"[message {row.message_id}] {row.content}"
        )

    return "\n".join(parts)

def build_group_documents(groups:pd.DataFrame):
    group_documents = (
        groups
        .groupby(
            ["source_id","source_name","root_message_id"],
            as_index=False
        )
        .apply(
            lambda thread: pd.Series({
                "content": build_thread_document(thread),

                "message_ids": thread["message_id"].tolist(),

                "media_paths": (
                    thread["media_path"]
                    .dropna()
                    .tolist()
                ),

                "message_date": thread["message_date"].min(),

                "message_count": len(thread)
            }),
            include_groups=False
        )
        .reset_index(drop=True)
    )

    group_documents["document_type"] = "group_thread"

    group_documents["document_id"] = (
        group_documents["source_id"].astype(str)
        + "_thread_"
        + group_documents["root_message_id"].astype(str)
    )

    return group_documents

def merge_documents(channel_documents:pd.DataFrame,group_documents:pd.DataFrame):
    channel_final = channel_documents.copy()

    channel_final["message_ids"] = channel_final["message_id"].apply(
        lambda x:[int(x)]
    )

    channel_final["media_paths"] = channel_final["media_path"].apply(
        lambda x:[] if pd.isna(x) else [x]
    )

    channel_final = channel_final[
        [
            "document_id",
            "document_type",
            "source_id",
            "source_name",
            "message_date",
            "content",
            "message_ids",
            "media_paths"
        ]
    ]

    group_final = group_documents[
        [
            "document_id",
            "document_type",
            "source_id",
            "source_name",
            "message_date",
            "content",
            "message_ids",
            "media_paths"
        ]
    ].copy()

    documents = pd.concat(
        [channel_final,group_final],
        ignore_index=True
    )

    return documents

def save_documents(documents:pd.DataFrame):
    output_dir = BASE_DIR / "data" / "processed"
    output_dir.mkdir(parents=True,exist_ok=True)

    output_path = output_dir / "documents.jsonl"

    documents.to_json(
        output_path,
        orient="records",
        lines=True,
        force_ascii=False
    )

    print(f"Saved: {output_path}")

def main():
    print("Loading database...")
    df = get_database(DB_PATH)
    print(f"Loaded {len(df):,} rows")

    print("Cleaning data...")
    df = cleaning_data(df)
    print(f"Remaining {len(df):,} rows")

    channels,groups = split_data(df)

    print(f"Channels: {len(channels):,}")
    print(f"Groups: {len(groups):,}")

    parent_map = build_parent_map(groups)

    groups = add_root_message(groups,parent_map)

    print("Building channel documents...")
    channel_documents = build_channel_documents(channels)

    group_documents = build_group_documents(groups)

    documents = merge_documents(channel_documents,group_documents)
    

    print(f"Final documents: {len(documents):,}")

    save_documents(documents)

    print("Done.")


if __name__ == "__main__":
    main()
