from typing import Any
from FlagEmbedding import BGEM3FlagModel
from tqdm import tqdm
import hashlib
import pymysql
import json
import numpy as np
import faiss
from config import DB_CONFIG

with open("novel_meta.json", "r", encoding="utf-8") as f:
    novel_meta = json.load(f)

title_to_pos = {m["title"].lower(): m["pos"] for m in novel_meta}
url_to_pos = {m["url"]: m["pos"] for m in novel_meta}
pos_to_url = {m["pos"]: m["url"] for m in novel_meta}
pos_to_title = {m["pos"]: m["title"] for m in novel_meta}

model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

with pymysql.connect(**DB_CONFIG) as conn:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT title, embedding, url
            FROM novels
            ORDER BY title
        """)
        rows = cur.fetchall()

titles = [r[0] for r in rows]
dim = 1024
embedding_store = np.vstack([
    np.frombuffer(r[1], dtype=np.float32, count=dim)
    for r in rows
]).astype(np.float32)
urls = [r[2] for r in rows]

def url_hash(url: str) -> str:
    """Generate SHA-256 hash for a URL."""
    return hashlib.sha256(url.encode("utf-8")).hexdigest()

def txt_factory(cursor, row) -> str:
    """
    Converts a database row into string format
    :param cursor: database cursor
    :param row: database row
    :return: row in string form
    """
    txt = []
    for i, col in enumerate(cursor.description):
        if row[i] is None:
            continue
        txt.append(f"{col[0]}: {row[i]}")
    return "\n".join(txt)

def upsert_neighbors(conn, rows):
    conn.cursor().executemany("""
    INSERT INTO novel_neighbors (
        source_url_hash,
        source_url,
        source_title,
        kind,
        rank,
        target_url_hash,
        target_url,
        target_title,
        score
    )
    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
    ON DUPLICATE KEY UPDATE
        target_url = VALUES(target_url),
        target_title = VALUES(target_title),
        score = VALUES(score)
    """, rows)

def extract_text() -> list[str]:
    """
    extracts text from database and stores it in a dictionary
    :return: Dict
    """
    with pymysql.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT title, author, genres, synopsis
                FROM novels
                ORDER BY title
            """)
            rows = cur.fetchall()

    novel_list = []
    for title, author, genres, synopsis in rows:
        parts = []
        if title:
            parts.append(f"title: {title}")
        if author:
            parts.append(f"author: {author}")
        if genres:
            parts.append(f"genres: {genres}")
        if synopsis:
            parts.append(f"synopsis: {synopsis}")
        novel_list.append("\n".join(parts))

    return novel_list

def embed_novels(novels: list[str], batch_size: int = 4, max_length: int = 2048) -> np.ndarray:
    """
    Using BGE-M3 Flag Model create vector embeddings of novels
    :param novels:a list of novel metadata in string format
    :param batch_size:
    :param max_length:
    :return:the embedded vectors in a np.ndarray
    """
    out = model.encode(
        novels,
        batch_size=batch_size,
        max_length=max_length,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )
    vec_novels = np.asarray(out["dense_vecs"], dtype=np.float32)
    return vec_novels

def faiss_index(embeddings: np.ndarray) -> faiss.IndexIVFPQ:
    """
    Creates a faiss index and stores it in novels_meta.json
    :param embeddings:
    :return:
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.maximum(norms, 1e-12)
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, "novels.faiss")

    with open("novel_meta.json", "w", encoding="utf-8") as f:
        json.dump(
            [{"pos": i, "title": titles[i], "url": urls[i]} for i in range(len(titles))],
            f,
            ensure_ascii=False,
            indent=2
        )
    return index

def query_faiss(index, title, k) -> tuple[Any, Any, Any]:
    """
    Query the FAISS index for the k closest and furthest titles
    :param index:FAISS index
    :param title:title undergoing similarity search
    :param k:number of titles to search for
    :return: k+1 closest titles, k farthest titles, title position
    """
    k += 1
    pos = title_to_pos.get(title.lower())
    xq = np.empty((index.d,), dtype=np.float32)
    index.reconstruct(pos, xq)
    return index.search(xq.reshape(1, -1), k), index.search(-xq.reshape(1, -1), k-1), pos

def main():
    novel_list = extract_text()
    vec_list = []
    chunk = 512

    # for i in tqdm(range(0, len(novel_list), chunk)):
    #     novel_vec = embed_novels(novel_list[i:i+chunk])
    #     vec_list.append(novel_vec)
    #
    # embeddings = np.vstack(vec_list)
    # dim = int(embeddings.shape[1])
    #
    # with pymysql.connect(**DB_CONFIG) as conn:
    #     with conn.cursor() as cur:
    #         cur.execute("SELECT url FROM novels ORDER BY title")
    #         urls_db = [r[0] for r in cur.fetchall()]
    #
    #         cur.executemany(
    #             """
    #             UPDATE novels
    #             SET embedding=%s, embedding_dim=%s, embedding_model=%s
    #             WHERE url=%s
    #             """,
    #             [(embeddings[i].tobytes(), dim, "BAAI/bge-m3", urls_db[i]) for i in range(len(urls_db))]
    #         )
    #     conn.commit()

    index = faiss_index(embedding_store)

    with pymysql.connect(**DB_CONFIG) as conn:
        index = faiss.read_index("novels.faiss")

        for title in title_to_pos:
            batch = []
            nearest, farthest, pos = query_faiss(index, title, 8)
            if nearest is None:
                continue

            D, I = nearest
            D2, I2 = farthest

            src_url = pos_to_url[pos]
            src_hash = url_hash(src_url)

            for rank, (score, hit_pos) in enumerate(zip(D[0], I[0]), start=1):
                if hit_pos == -1 or hit_pos == pos:
                    continue
                tgt_url = pos_to_url[hit_pos]
                batch.append((
                    src_hash,
                    src_url,
                    title,
                    "nearest",
                    rank,
                    url_hash(tgt_url),
                    tgt_url,
                    pos_to_title[hit_pos],
                    float(score),
                ))

            for rank, (score, hit_pos) in enumerate(zip(D2[0], I2[0]), start=1):
                if hit_pos == -1 or hit_pos == pos:
                    continue
                tgt_url = pos_to_url[hit_pos]
                batch.append((
                    src_hash,
                    src_url,
                    title,
                    "farthest",
                    rank,
                    url_hash(tgt_url),
                    tgt_url,
                    pos_to_title[hit_pos],
                    float(-score),
                ))

            upsert_neighbors(conn, batch)
            conn.commit()
            print(f"novel uploaded {title}")

if __name__ == "__main__":
    main()
