import pymysql
import numpy as np
from tqdm import tqdm
from FlagEmbedding import BGEM3FlagModel
from config import DB_CONFIG
import json

MODEL_NAME = "../models/checkpoint-3600v3" # "BAAI/bge-m3"
BATCH_SIZE = 16
MAX_LENGTH = 256


def load_novels():
    # with open("bge_test.jsonl", "r", encoding="utf-8") as f:
    #     for line in f:
    #         obj = json.loads(line)
    #
    #         q = obj["query"]
    #
    #         # Find first newline without creating split list
    #         nl = q.find("\n")
    #         first_line = q if nl == -1 else q[:nl]
    #
    #         # "Title:" is 6 characters
    #         if first_line.startswith("Title:"):
    #             allowed_titles.add(first_line[6:].lstrip())
    # placeholders = ",".join(["%s"] * len(allowed_titles))
    with pymysql.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT novel_id, novel_meta
                FROM novels
                WHERE novel_meta IS NOT NULL 
                ORDER BY novel_id
            """)
            rows = cur.fetchall()

    ids = [int(r[0]) for r in rows]
    texts = [r[1] or "" for r in rows]

    return ids, texts


def embed_batches(model, texts):
    all_vecs = []

    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Embedding"):
        batch = texts[i:i + BATCH_SIZE]

        out = model.encode(
            batch,
            batch_size=BATCH_SIZE,
            max_length=MAX_LENGTH,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )

        vecs = np.asarray(out["dense_vecs"], dtype=np.float32)
        all_vecs.append(vecs)

    embeddings = np.vstack(all_vecs)

    # Normalize for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.maximum(norms, 1e-12)

    return embeddings


def persist_embeddings(ids, embeddings):
    dim = embeddings.shape[1]

    with pymysql.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            cur.executemany(
                """
                UPDATE novels
                SET embedding=%s,
                    embedding_dim=%s,
                    embedding_model=%s
                WHERE novel_id=%s
                """,
                [
                    (
                        embeddings[i].tobytes(),
                        dim,
                        MODEL_NAME,
                        ids[i]
                    )
                    for i in range(len(ids))
                ]
            )
        conn.commit()


def main():
    print("Loading novels...")
    ids, texts = load_novels()

    print(f"Loaded {len(ids)} novels.")

    print("Loading model...")
    model = BGEM3FlagModel(MODEL_NAME, use_fp16=True, device="mps")
    print("Model path:", model.model_name_or_path)

    print("Embedding...")
    embeddings = embed_batches(model, texts)

    print("Persisting to database...")
    persist_embeddings(ids, embeddings)

    print("Saving local .npy backup...")
    np.save("embeddings.npy", embeddings)

    print("Done.")


if __name__ == "__main__":
    main()