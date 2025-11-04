"""PDF Processing and Index Building

The foundation of the whole system. I spent way too much time optimizing this part,
but it was worth it - the chunking strategy here makes or breaks the retrieval quality.

My approach:
- Extract text page by page (preserves context better than full-doc extraction)  
- 500-word chunks with 100-word overlap (found this balance through trial and error)
- sentence-transformers because they're lightweight and actually pretty good
- FAISS IndexFlatIP - simple but blazingly fast for this use case

Run this first before anything else:
    python build_embeddings.py --pdfs "doc1.pdf,doc2.pdf" --out_dir data
"""
import json
import argparse
from pathlib import Path
from typing import List, Tuple

try:
    from PyPDF2 import PdfReader
except Exception:
    raise ImportError("PyPDF2 is required. Install with: pip install PyPDF2")

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss  # type: ignore


def extract_pages_text(pdf_path: Path) -> List[Tuple[int, str]]:
    reader = PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        pages.append((i + 1, text))
    return pages


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    if not text:
        return []
    words = text.split()
    if len(words) <= chunk_size:
        return [" ".join(words)]
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start = end - overlap
    return chunks


def l2_normalize(x: np.ndarray) -> np.ndarray:
    # normalize rows
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def build_index(embeddings: np.ndarray):  # type: ignore
    d = embeddings.shape[1]
    # FAISS requires float32 arrays
    embeddings = embeddings.astype(np.float32)
    # use inner product on normalized vectors to get cosine similarity
    index = faiss.IndexFlatIP(d)  # type: ignore
    index.add(embeddings)  # type: ignore
    return index


def main(args):
    workspace = Path(args.workspace)
    pdf_paths = []
    if args.pdfs:
        for p in args.pdfs.split(","):
            p = p.strip()
            candidate = workspace / p if not Path(p).is_absolute() else Path(p)
            if candidate.exists():
                pdf_paths.append(candidate)
            else:
                print(f"Warning: PDF not found: {candidate}")
    else:
        # find PDFs in workspace
        for p in workspace.glob("*.pdf"):
            pdf_paths.append(p)

    if not pdf_paths:
        print("No PDFs found. Provide --pdfs or put PDFs in workspace root.")
        return

    model_name = args.model_name
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    chunks = []
    metadata = []

    for pdf_path in pdf_paths:
        print(f"Processing: {pdf_path.name}")
        pages = extract_pages_text(pdf_path)
        for page_num, text in pages:
            page_chunks = chunk_text(text, chunk_size=args.chunk_size, overlap=args.overlap)
            for i, ch in enumerate(page_chunks):
                chunks.append(ch)
                metadata.append({
                    "source": pdf_path.name,
                    "page": page_num,
                    "chunk_id": len(metadata),
                    "text_snippet": ch[:400]
                })

    if not chunks:
        print("No text chunks created from PDFs.")
        return

    print(f"Encoding {len(chunks)} chunks...")
    embeddings = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True, batch_size=64)
    embeddings = embeddings.astype(np.float32)  # Ensure float32 for FAISS
    embeddings = l2_normalize(embeddings)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Building FAISS index...")
    index = build_index(embeddings)

    faiss_index_path = out_dir / "faiss.index"
    print(f"Saving FAISS index to {faiss_index_path}")
    faiss.write_index(index, str(faiss_index_path))  # type: ignore

    metadata_path = out_dir / "metadata.json"
    print(f"Saving metadata to {metadata_path}")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    info = {
        "num_chunks": len(chunks),
        "model_name": model_name,
        "chunk_size": args.chunk_size,
        "overlap": args.overlap
    }
    info_path = out_dir / "index_info.json"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    # optional: save raw chunks separately for retrieval convenience
    chunks_path = out_dir / "chunks.jsonl"
    print(f"Saving chunks to {chunks_path}")
    with open(chunks_path, "w", encoding="utf-8") as f:
        for m, ch in zip(metadata, chunks):
            entry = {**m, "text": ch}
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default=".", help="Workspace root where PDFs live")
    parser.add_argument("--pdfs", help="Comma-separated list of PDF filenames in workspace to index")
    parser.add_argument("--out_dir", default="data", help="Output directory for index and metadata")
    parser.add_argument("--model_name", default="all-MiniLM-L6-v2", help="SentenceTransformers model")
    parser.add_argument("--chunk_size", type=int, default=500, help="Chunk size in words")
    parser.add_argument("--overlap", type=int, default=100, help="Chunk overlap in words")
    args = parser.parse_args()
    main(args)
