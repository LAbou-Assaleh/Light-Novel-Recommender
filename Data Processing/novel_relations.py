import json
import pymysql
from collections import defaultdict
from tqdm import tqdm
from config import DB_CONFIG


def build_relations():
    with pymysql.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT items_json
                FROM recommendation_lists
            """)
            rows = cur.fetchall()

        total_lists = len(rows)

        # title -> related_title -> count
        relations = defaultdict(lambda: defaultdict(int))

        # title -> number of lists it appears in
        appearance_count = defaultdict(int)

        for (items_json,) in tqdm(rows, desc="Processing recommendation lists"):
            if not items_json:
                continue

            items = json.loads(items_json)
            titles = {item["Title"] for item in items if "Title" in item}

            for title in titles:
                appearance_count[title] += 1

            for src in titles:
                for tgt in titles:
                    if src == tgt:
                        continue
                    relations[src][tgt] += 1

        with conn.cursor() as cur:
            for src in relations:
                for tgt in relations[src]:
                    cur.execute("""
                        INSERT INTO novel_relations (title, related_titles, occurrence)
                        VALUES (%s, %s, %s)
                    """, [src, tgt, relations[src][tgt]])
                    print(f"Inserted {src, tgt, relations[src][tgt]} novel relations")
        conn.commit()



if __name__ == "__main__":
    build_relations()
# import json
# import sqlite3
# from collections import defaultdict
# from tqdm import tqdm
#
# DB_PATH = "novels.db"
#
#
# def init_sqlite(db_path: str):
#     with sqlite3.connect(db_path) as conn:
#         cur = conn.cursor()
#         cur.execute("""
#             CREATE TABLE IF NOT EXISTS novel_relations (
#                 title TEXT PRIMARY KEY,
#                 related_titles JSON,
#                 percent REAL
#             )
#         """)
#
#
# def build_and_upsert_relations():
#     init_sqlite(DB_PATH)
#
#     with sqlite3.connect(DB_PATH) as conn:
#         cur = conn.cursor()
#
#         cur.execute("""
#             SELECT items_json
#             FROM recommendation_lists
#         """)
#         rows = cur.fetchall()
#
#         total_lists = len(rows)
#
#         relations = defaultdict(lambda: defaultdict(int))
#         appearance_count = defaultdict(int)
#
#         for (items_json,) in tqdm(rows, desc="Processing recommendation lists"):
#             if not items_json:
#                 continue
#
#             items = json.loads(items_json)
#             titles = {item["Title"] for item in items if "Title" in item}
#
#             for title in titles:
#                 appearance_count[title] += 1
#
#             for src in titles:
#                 for tgt in titles:
#                     if src == tgt:
#                         continue
#                     relations[src][tgt] += 1
#
#         upsert_rows = [
#             (
#                 src,
#                 json.dumps(dict(related), ensure_ascii=False),
#                 (appearance_count[src] / total_lists) * 100
#             )
#             for src, related in relations.items()
#         ]
#
#         cur.executemany("""
#             INSERT INTO novel_relations (title, related_titles, percent)
#             VALUES (?, ?, ?)
#             ON CONFLICT(title) DO UPDATE SET
#                 related_titles = excluded.related_titles,
#                 percent = excluded.percent
#         """, upsert_rows)
#
#     print(f"Upserted {len(upsert_rows)} novel relations")
#
#
# if __name__ == "__main__":
#     build_and_upsert_relations()
#
# import pymysql
# import sqlite3
# from tqdm import tqdm
# from config import DB_CONFIG
#
# SQLITE_PATH = "novels.db"
#
#
# def init_sqlite():
#     with sqlite3.connect(SQLITE_PATH) as conn:
#         cur = conn.cursor()
#         cur.execute("""
#             CREATE TABLE IF NOT EXISTS novels (
#                 url_hash TEXT PRIMARY KEY,
#                 url TEXT UNIQUE,
#                 title TEXT,
#                 author TEXT,
#                 status TEXT,
#                 time_of_last_update TEXT,
#                 genres TEXT,
#                 rating REAL,
#                 num_rating INTEGER,
#                 synopsis TEXT,
#                 scraped_at TEXT,
#                 embedding BLOB,
#                 embedding_dim INTEGER,
#                 embedding_model TEXT
#             )
#         """)
#
#
# def migrate_novels():
#     init_sqlite()
#
#     # --- read from MariaDB ---
#     with pymysql.connect(**DB_CONFIG) as maria_conn:
#         with maria_conn.cursor() as cur:
#             cur.execute("""
#                 SELECT
#                     url_hash,
#                     url,
#                     title,
#                     author,
#                     status,
#                     time_of_last_update,
#                     genres,
#                     rating,
#                     num_rating,
#                     synopsis,
#                     scraped_at,
#                     embedding,
#                     embedding_dim,
#                     embedding_model
#                 FROM novels
#             """)
#             rows = cur.fetchall()
#
#     # --- write to SQLite ---
#     with sqlite3.connect(SQLITE_PATH) as sqlite_conn:
#         cur = sqlite_conn.cursor()
#
#         cur.executemany("""
#             INSERT INTO novels (
#                 url_hash,
#                 url,
#                 title,
#                 author,
#                 status,
#                 time_of_last_update,
#                 genres,
#                 rating,
#                 num_rating,
#                 synopsis,
#                 scraped_at,
#                 embedding,
#                 embedding_dim,
#                 embedding_model
#             )
#             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
#             ON CONFLICT(url_hash) DO UPDATE SET
#                 url = excluded.url,
#                 title = excluded.title,
#                 author = excluded.author,
#                 status = excluded.status,
#                 time_of_last_update = excluded.time_of_last_update,
#                 genres = excluded.genres,
#                 rating = excluded.rating,
#                 num_rating = excluded.num_rating,
#                 synopsis = excluded.synopsis,
#                 scraped_at = excluded.scraped_at,
#                 embedding = excluded.embedding,
#                 embedding_dim = excluded.embedding_dim,
#                 embedding_model = excluded.embedding_model
#         """, tqdm(rows, desc="Migrating novels"))
#
#     print(f"Migrated {len(rows)} novels to SQLite")
#
#
# if __name__ == "__main__":
#     migrate_novels()