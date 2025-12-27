import os, glob, json
import numpy as np
import faiss
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_DIR = "data"
INDEX_DIR = "index"

def load_texts():
    texts, sources = [], []
    for path in glob.glob(os.path.join(DATA_DIR, "*.txt")):
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content:
                texts.append(content)
                sources.append(os.path.basename(path))
    return texts, sources

def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
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

def main():
    load_dotenv()

    texts, sources = load_texts()
    if not texts:
        raise ValueError("No content found in data/*.txt")

    docs, meta = [], []
    for src_text, src_name in zip(texts, sources):
        for i, ch in enumerate(chunk_text(src_text)):
            docs.append(ch)
            meta.append({"source": src_name, "chunk_id": i})

    print(f"Total chunks: {len(docs)}")

    # TF-IDF embeddings
    vectorizer = TfidfVectorizer(max_features=768)
    X = vectorizer.fit_transform(docs).toarray().astype("float32")

    faiss.normalize_L2(X)
    index = faiss.IndexFlatIP(X.shape[1])
    index.add(X)

    os.makedirs(INDEX_DIR, exist_ok=True)
    faiss.write_index(index, os.path.join(INDEX_DIR, "faiss.index"))
    np.save(os.path.join(INDEX_DIR, "embeddings.npy"), X)
    with open(os.path.join(INDEX_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"docs": docs, "meta": meta}, f, indent=2)

    print("Index built and saved!")

if __name__ == "__main__":
    main()
