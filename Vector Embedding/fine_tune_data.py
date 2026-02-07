import json
import random
import gzip
import pymysql
from tqdm import tqdm
from config import DB_CONFIG

TRAIN_RATIO = 0.85
TEST_RATIO = 0.05
PRIVATE_RATIO = 0.10

RANDOM_NEGATIVES = 3      # per query
SEED = 42

PROMPT = "Which novel is most similar to:"
TYPE = "normal"

random.seed(SEED)
def create_fine_tune_data():
    """
    Extracts novel relationships and formats it in jsonl files for future fine tuning
    :return:
    """
    train_f = gzip.open("bge_train.jsonl.gz", "wt", encoding="utf-8")
    test_f = gzip.open("bge_test.jsonl.gz", "wt", encoding="utf-8")
    private_f = gzip.open("bge_private.jsonl.gz", "wt", encoding="utf-8")
    conn = pymysql.connect(**DB_CONFIG)

    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT title, novel_meta
                FROM novels
                WHERE novel_meta IS NOT NULL
                ORDER BY title
            """)
            novel_meta = {title: meta for title, meta in cur.fetchall()}

            print(f"Loaded novel_meta for {len(novel_meta)} novels")
            cur.execute(f"""
                SELECT title, related_titles, relative_percent_occurrence
                FROM novel_relations
                ORDER BY title, related_titles
            """)

            relations = {}
            for title, rel, p in cur.fetchall():
                relations.setdefault(title, []).append((rel, float(p)))

            print(f"Loaded relations for {len(relations)} source novels")
        titles = list(relations.keys())
        random.shuffle(titles)

        for title in tqdm(titles, desc="Processing novels", unit="novel"):
            if title not in novel_meta:
                continue

            rels = relations[title]
            strong_pos = [(r, p) for r, p in rels if p >= 0.60]
            pos = [(r, p) for r, p in rels if 0.25 <= p < 0.60]
            hard_neg = [(r, p) for r, p in rels if 0.05 <= p < 0.25]

            if not (strong_pos or pos):
                continue

            positives = strong_pos + pos
            negatives = hard_neg.copy()

            pool = [
                t for t in novel_meta.keys()
                if t != title and t not in {r for r, _ in positives+negatives}
            ]

            if pool:
                negatives += [(t, 0.02) for t in random.sample(pool, k=min(RANDOM_NEGATIVES, len(pool)))]
            pos_texts, pos_scores = [], []
            for r, p in positives:
                if r in novel_meta:
                    pos_texts.append(novel_meta[r])
                    pos_scores.append(round(min(1.0, max(0.5, p)), 3))

            neg_texts, neg_scores = [], []
            for r, p in negatives:
                if r in novel_meta:
                    neg_texts.append(novel_meta[r])
                    neg_scores.append(round(min(0.3, max(0.01, p)), 3))

            if not pos_texts or not neg_texts:
                continue

            example = {
                "query": novel_meta[title],
                "pos": pos_texts,
                "neg": neg_texts,
                "pos_scores": pos_scores,
                "neg_scores": neg_scores,
                "prompt": PROMPT,
                "type": TYPE,
            }
            r = random.random()
            if r < TRAIN_RATIO:
                target_f = train_f
            elif r < TRAIN_RATIO + TEST_RATIO:
                target_f = test_f
            else:
                target_f = private_f
            target_f.write(json.dumps(example, ensure_ascii=False)+"\n")

    finally:
        conn.close()
        train_f.close()
        test_f.close()
        private_f.close()

    print("Finished creating BGE-M3 JSONL datasets")
if __name__ == "__main__":
    create_fine_tune_data()