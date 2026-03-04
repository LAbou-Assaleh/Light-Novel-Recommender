import faiss
import numpy as np
import pymysql
from config import DB_CONFIG


OLD_INDEX_PATH = "old_novels.faiss"
NEW_INDEX_PATH = "novels.faiss"
EMBED_DIM = 1024

def load_old_embeddings():
    with pymysql.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT novel_id, old_embedding
                FROM novels
                WHERE old_embedding IS NOT NULL
                ORDER BY novel_id
            """)
            rows = cur.fetchall()

    if not rows:
        raise ValueError("No embeddings found.")

    ids = []
    vectors = []

    for nid, blob in rows:
        vec = np.frombuffer(blob, dtype=np.float32, count=EMBED_DIM)
        ids.append(nid)
        vectors.append(vec)

    embeddings = np.vstack(vectors).astype(np.float32)
    ids = np.asarray(ids, dtype=np.int64)

    return embeddings, ids

def load_new_embeddings():
    with pymysql.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT novel_id, embedding
                FROM novels
                WHERE embedding IS NOT NULL
                ORDER BY novel_id
            """)
            rows = cur.fetchall()

    if not rows:
        raise ValueError("No embeddings found.")

    ids = []
    vectors = []

    for nid, blob in rows:
        vec = np.frombuffer(blob, dtype=np.float32, count=EMBED_DIM)
        ids.append(nid)
        vectors.append(vec)

    embeddings = np.vstack(vectors).astype(np.float32)
    ids = np.asarray(ids, dtype=np.int64)

    return embeddings, ids

def normalize(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.maximum(norms, 1e-12)


def build_index(embeddings: np.ndarray, ids: np.ndarray) -> faiss.Index:
    dim = embeddings.shape[1]

    base = faiss.IndexFlatIP(dim)  # cosine via inner product
    index = faiss.IndexIDMap(base)

    index.add_with_ids(embeddings, ids)
    return index


def main():
    print("Loading embeddings...")
    old_embeddings, old_ids = load_old_embeddings()
    new_embeddings, new_ids = load_new_embeddings()

    print("Normalizing...")
    old_embeddings = normalize(old_embeddings)
    new_embeddings = normalize(new_embeddings)

    print("Building FAISS index...")
    old_index = build_index(old_embeddings, old_ids)
    new_index = build_index(new_embeddings, new_ids)

    print(f"New Index size: {new_index.ntotal}")
    print(f"Old Index size: {old_index.ntotal}")

    print("Saving index...")
    faiss.write_index(new_index, NEW_INDEX_PATH)
    faiss.write_index(old_index, OLD_INDEX_PATH)

    print("Done.")


if __name__ == "__main__":
    main()