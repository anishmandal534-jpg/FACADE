import os
import re

from pypdf import PdfReader
from qdrant_client.models import PointStruct

from .vector_db import client, COLLECTION_NAME, init_vector_db
from .embeddings import embed_texts


def extract_text_from_file(file_path: str) -> str:
    text = ""

    if not file_path:
        return text

    lower_path = file_path.lower()

    if lower_path.endswith(".pdf"):
        reader = PdfReader(file_path)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"

    elif lower_path.endswith(".txt") or lower_path.endswith(".md"):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

    return text


def process_and_store_document(
    file_path: str = None,
    persona: str = "My Personal Twin",
    direct_text: str = None,
):
    if direct_text:
        raw_text = direct_text
    elif file_path:
        raw_text = extract_text_from_file(file_path)
    else:
        raw_text = ""

    if not raw_text:
        return 0

    init_vector_db()

    raw_text = re.sub(r"\s+", " ", raw_text).strip()
    if not raw_text:
        return 0

    chunks = []
    words = raw_text.split()

    for i in range(0, len(words), 300):
        chunk = " ".join(words[i:i + 300]).strip()
        if chunk:
            chunks.append(chunk)

    if not chunks:
        return 0

    embeddings = embed_texts(chunks)

    source_name = (
        os.path.basename(file_path)
        if file_path
        else f"{persona}_profile"
    )

    points = []

    for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        point_id = hash(
            f"{source_name}_{idx}_{persona}"
        ) & 0x7FFFFFFF

        points.append(
            PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "text": chunk,
                    "source": source_name,
                    "persona": persona,
                },
            )
        )

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
    )

    return len(points)
