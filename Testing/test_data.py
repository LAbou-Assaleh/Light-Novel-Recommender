import json
import random
import pymysql
from tqdm import tqdm
from config import DB_CONFIG
import csv
import sys

allowed_titles = set()
N = 1_000

print("Extracting Titles")

with open("./all-data/bge_private.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        q = obj["query"]

        nl = q.find("\n")
        first_line = q if nl == -1 else q[:nl]

        if first_line.startswith("Title:"):
            allowed_titles.add(first_line[6:].lstrip())
with open("./all-data/bge_test.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        q = obj["query"]

        nl = q.find("\n")
        first_line = q if nl == -1 else q[:nl]

        if first_line.startswith("Title:"):
            allowed_titles.add(first_line[6:].lstrip())
with open("./all-data/bge_train.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        q = obj["query"]

        nl = q.find("\n")
        first_line = q if nl == -1 else q[:nl]

        if first_line.startswith("Title:"):
            allowed_titles.add(first_line[6:].lstrip())
titles_tuple = tuple(sorted(allowed_titles))

with open("allowed_titles.txt", "w", encoding="utf-8") as f:
    f.write(repr(titles_tuple))

print(f"Stored {len(titles_tuple)} titles.")
print("Extracted Titles")


def create_test_data(N=10000):
    """
    Extract novel relationships and format them into JSONL test file.
    """

    output_path = f"bge_test_embeddings{N}.jsonl"
    test_f = open(output_path, "w", encoding="utf-8")

    conn = pymysql.connect(**DB_CONFIG)

    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT title, novel_id
                FROM novels
                WHERE novel_id IS NOT NULL
                ORDER BY title
            """)
            novel_id = {title: id for title, id in cur.fetchall()}

            print(f"Loaded novel_id for {len(novel_id)} novels")
            placeholders = ",".join(["%s"] * len(allowed_titles))

            cur.execute(f"""
                SELECT title, related_titles, relative_percent_occurrence
                FROM (
                    SELECT
                        title,
                        related_titles,
                        relative_percent_occurrence,
                        ROW_NUMBER() OVER (
                            PARTITION BY title
                            ORDER BY relative_percent_occurrence DESC
                        ) AS rn
                    FROM novel_relations
                    WHERE title IN ({placeholders})
                ) t
                WHERE rn <= {N}
                ORDER BY title, relative_percent_occurrence DESC;
            """, tuple(allowed_titles))

            relations = {}
            for title, rel, p in cur.fetchall():
                relations.setdefault(title, []).append(rel)

            print(f"Loaded relations for {len(relations)} source novels")
        titles = list(relations.keys())

        for title in tqdm(titles, desc="Processing novels", unit="novel"):

            if title not in novel_id:
                continue

            rels = [novel_id[name] for name in relations[title] if name in novel_id]

            if len(rels) < N / 2:
                continue

            test = {
                "query": novel_id[title],
                "results": rels
            }

            test_f.write(json.dumps(test, ensure_ascii=False) + "\n")

    finally:
        conn.close()
        test_f.close()

    print(f"Finished creating {output_path}")


if __name__ == "__main__":
    create_test_data(N)
