import json
from pathlib import Path

import faiss
import fitz
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

# --------------------------------
# INITIALIZATION
# --------------------------------

BASE_DIR = Path(__file__).resolve().parent

PDF_DIR = BASE_DIR / "knowledge" / "pdfs"
PROCESSED_DIR = BASE_DIR / "knowledge" / "processed"

INDEX_PATH = PROCESSED_DIR / "faiss.index"
CHUNKS_PATH = PROCESSED_DIR / "chunks.json"

EMBEDDING_MODEL = "text-embedding-3-small"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 75

load_dotenv(BASE_DIR / ".env")

client = OpenAI()


# --------------------------------
# STEP 1: EXTRACT TEXT FROM PDF
# --------------------------------

def extract_pdf_pages(pdf_path: Path):

    pages = []

    with fitz.open(pdf_path) as pdf:

        for page_number, page in enumerate(pdf, start=1):

            text = page.get_text("text").strip()

            if text:
                pages.append({
                    "text": text,
                    "page": page_number
                })

    return pages


# --------------------------------
# STEP 2: SPLIT TEXT INTO CHUNKS
# --------------------------------

def split_into_chunks(text, chunk_size=CHUNK_SIZE,
                      overlap=CHUNK_OVERLAP):

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be non-negative and smaller than chunk_size"
        )

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start = end - overlap

    return chunks


# --------------------------------
# STEP 3: PREPARE CHUNKS + METADATA
# --------------------------------

def process_pdf(pdf_path: Path):

    print(f"Processing: {pdf_path.name}")

    pages = extract_pdf_pages(pdf_path)

    document_chunks = []

    for page_data in pages:

        page_number = page_data["page"]
        page_text = page_data["text"]

        chunks = split_into_chunks(page_text)

        for chunk_index, chunk_text in enumerate(chunks):

            document_chunks.append({
                "chunk_id": (
                    f"{pdf_path.stem}_p{page_number}_c{chunk_index}"
                ),
                "text": chunk_text,
                "source": pdf_path.name,
                "page": page_number,
                "chunk_index": chunk_index
            })

    return document_chunks


# --------------------------------
# STEP 4: GENERATE EMBEDDINGS
# --------------------------------

def generate_embeddings(texts, batch_size=100):

    all_embeddings = []

    for start in range(0, len(texts), batch_size):

        batch = texts[start:start + batch_size]

        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch
        )

        results = sorted(
            response.data,
            key=lambda item: item.index
        )

        all_embeddings.extend(
            item.embedding for item in results
        )

    return all_embeddings


# --------------------------------
# STEP 5: BUILD AND SAVE FAISS INDEX
# --------------------------------

def build_and_save_index(all_chunks):

    if not all_chunks:
        raise ValueError(
            "No text was extracted. Check your PDF files."
        )

    texts = [
        chunk["text"]
        for chunk in all_chunks
    ]

    embeddings = generate_embeddings(texts)

    vectors = np.array(
        embeddings,
        dtype="float32"
    )

    faiss.normalize_L2(vectors)

    dimension = vectors.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(vectors)

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    faiss.write_index(
        index,
        str(INDEX_PATH)
    )

    CHUNKS_PATH.write_text(
        json.dumps(
            all_chunks,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    print("\nIngestion completed successfully!")
    print(f"Total chunks: {len(all_chunks)}")
    print(f"Total vectors: {index.ntotal}")
    print(f"Vector dimensions: {dimension}")


# --------------------------------
# MAIN EXECUTION
# --------------------------------

def main():

    pdf_files = list(PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in {PDF_DIR}"
        )

    all_chunks = []

    for pdf_path in pdf_files:

        document_chunks = process_pdf(pdf_path)

        all_chunks.extend(document_chunks)

    build_and_save_index(all_chunks)


if __name__ == "__main__":
    main()