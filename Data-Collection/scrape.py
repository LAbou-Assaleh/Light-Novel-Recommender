import os
import sqlite3
from urllib.parse import urljoin
from seleniumbase import SB
from bs4 import BeautifulSoup

os.environ["DISABLE_COLORS"] = "1"

startpage = 568
start_url = f"https://novelbin.com/sort/latest?page={startpage}"

db_file = "novelbin_latest_novels-Dec-26-2025.sqlite"
max_pages = None  # int or None
headless = True #keep true to run headless

def init_db() -> None:
    """
    Initialize the SQLite database schema.

    Creates the required tables and indexes if they do not already exist.
    Enables WAL mode and foreign key enforcement for the connection.
    Returns:
        None.
    """
    with sqlite3.connect(db_file) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.execute("""
        CREATE TABLE IF NOT EXISTS novels (
            url TEXT PRIMARY KEY,
            title TEXT,
            author TEXT,
            status TEXT,
            last_completed_chapter TEXT,
            time_of_last_update TEXT,
            genres TEXT,
            rating REAL,
            num_rating INTEGER,
            synopsis TEXT,
            scraped_at TEXT DEFAULT (datetime('now'))
        );
        """)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS novel_tags (
            url TEXT NOT NULL,
            tag TEXT NOT NULL,
            PRIMARY KEY (url, tag),
            FOREIGN KEY (url) REFERENCES novels(url) ON DELETE CASCADE
        );
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_novels_title ON novels(title);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_novels_author ON novels(author);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tags_tag ON novel_tags(tag);")


def url_exists(conn: sqlite3.Connection, url: str) -> bool:
    """
    Check whether a novel URL already exists in the database.

    Args:
        conn: An open SQLite connection to the novels database.
        url: The novel detail page URL to check.

    Returns:
        True if the URL exists in the "novels" table, otherwise False.
    """
    row = conn.execute("SELECT 1 FROM novels WHERE url = ? LIMIT 1;", (url,)).fetchone()
    return row is not None


def safe_float(x) -> float | None:
    """
    Converts a value to a float if possible.

    Args:
        x: The value to convert.

    Returns:
        The value as a float, or None if the conversion fails.
    """
    try:
        return float(x)
    except Exception:
        return None


def safe_int(x) -> int | None:
    """
    Convert a value to int if possible.

    Args:
        x: The value to convert.

    Returns:
        The value as an int, or None if conversion fails.
    """
    try:
        return int(x)
    except Exception:
        return None


def upsert_novel(conn: sqlite3.Connection, novel: dict) -> None:
    """
    Insert a novel into the database or update it if it already exists.

    Uses the novel's URL as the primary key and replaces the novel's tags
    in the "novel_tags" table with the tags provided in the record.

    Args:
        conn: An open SQLite connection to the novels database.
        novel: A novel record dictionary with keys such as:
            - "URL" (required)
            - "Title"
            - "Author"
            - "Status"
            - "Last Completed Chapter"
            - "Time of Last Update"
            - "Genres", "Rating"
            - "Num Rating"
            - "Synopsis"
            - "Tags" (list[str])

    Returns:
        None.
    """
    conn.execute("""
    INSERT INTO novels (
        url, title, author, status, last_completed_chapter,
        time_of_last_update, genres, rating, num_rating, synopsis, scraped_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
    ON CONFLICT(url) DO UPDATE SET
        title=excluded.title,
        author=excluded.author,
        status=excluded.status,
        last_completed_chapter=excluded.last_completed_chapter,
        time_of_last_update=excluded.time_of_last_update,
        genres=excluded.genres,
        rating=excluded.rating,
        num_rating=excluded.num_rating,
        synopsis=excluded.synopsis,
        scraped_at=datetime('now');
    """, (
        novel["URL"],
        novel.get("Title"),
        novel.get("Author"),
        novel.get("Status"),
        novel.get("Last Completed Chapter"),
        novel.get("Time of Last Update"),
        novel.get("Genres"),
        safe_float(novel.get("Rating")),
        safe_int(novel.get("Num Rating")),
        novel.get("Synopsis"),
    ))

    conn.execute("DELETE FROM novel_tags WHERE url = ?;", (novel["URL"],))
    tags = novel.get("Tags") or []
    conn.executemany(
        "INSERT OR IGNORE INTO novel_tags(url, tag) VALUES (?, ?);",
        [(novel["URL"], t) for t in tags]
    )


def get_novel_urls_from_page(html: str) -> list[str]:
    """
    Parse novel URLs from the "latest novels" listing page HTML.

    Extracts each novel detail page link from the listing and returns a list of
    absolute URLs.

    Args:
        html: HTML content of the latest novels page.

    Returns:
        A list of absolute novel URLs found on the page.
    """
    soup = BeautifulSoup(html, "html.parser")
    urls: list[str] = []
    rows = soup.select(".list-novel .row")
    for row in rows:
        link = row.select_one("h3.novel-title a")
        if link and link.get("href"):
            full_url = urljoin(start_url, link["href"])
            urls.append(full_url)
    return urls

def go_to_next_page(sb, page_num) -> bool:
    """
    Navigate to the next "latest novels" listing page.

    Args:
        sb:SeleniumBase browser session object.
        page_num:The "latest novels" listing page number being accessed.

    Returns:
        True if the page is successfully accessed otherwise False.
    """
    try:
        sb.open(f"https://novelbin.com/sort/latest?page={page_num}")
        return True
    except Exception:
        return False

def scrape_novel_with_bs4(html, url):
    """
    Extracts novel data using BeautifulSoup4 from the pages HTML.

    Args:
        html:HTML of the novel detail page.
        url:Absolute URL of the novel detail page.

    Returns:
        Dictonary containg the novels metadata
    """
    """Extract novel data using BeautifulSoup from detail page HTML."""
    soup = BeautifulSoup(html, "html.parser")

    def get_og(property_name):
        tag = soup.find("meta", property=property_name)
        return tag.get("content", "N/A") if tag else "N/A"

    title = get_og("og:novel:novel_name")
    if title == "N/A":
        title_tag = soup.find("meta", property="og:title")
        if title_tag:
            title = title_tag.get("content", "N/A").replace(" - Read ... - Novel Bin", "").strip()
        else:
            title_elem = soup.select_one("h3.title") or soup.select_one("h1")
            title = title_elem.get_text(strip=True) if title_elem else "N/A"

    author = get_og("og:novel:author")
    status = get_og("og:novel:status")
    last_chapter = get_og("og:novel:lastest_chapter_name")
    update_time = get_og("og:novel:update_time")
    genres = get_og("og:novel:genre")

    rating_el = soup.select_one('span[itemprop="ratingValue"]')
    rating = rating_el.get_text(strip=True) if rating_el else None

    count_el = soup.select_one('span[itemprop="reviewCount"]')
    num_rating = count_el.get_text(strip=True).replace(",", "") if count_el else None

    desc = soup.select_one('div.desc-text[itemprop="description"]')
    synopsis = desc.get_text(" ", strip=True) if desc else "N/A"
    synopsis = synopsis.replace("\xa0", " ")

    tags = []
    tag_container = soup.select_one(".tag-container")
    if tag_container:
        tag_links = tag_container.find_all("a")
        tags = [t.get_text(strip=True) for t in tag_links if t.get_text(strip=True)]

    return {
        "Title": title,
        "Author": author,
        "Status": status,
        "Last Completed Chapter": last_chapter,
        "Time of Last Update": update_time,
        "Genres": genres,
        "Tags": tags,
        "URL": url,
        "Rating": rating,
        "Num Rating": num_rating,
        "Synopsis": synopsis
    }

def main():
    init_db()
    page_count = startpage - 1

    with sqlite3.connect(db_file) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")

        with SB(uc=True, headless=headless) as sb:
            sb.open(start_url)

            while True:
                page_count += 1
                print(f"\n--- Scraping Page {page_count} ---")

                html = sb.get_page_source()
                novel_urls = get_novel_urls_from_page(html)
                print(f"Found {len(novel_urls)} novels on page {page_count}")
                if len(novel_urls) == 0:
                    break

                for url in novel_urls:
                    # Skip opening detail page if already in DB (fast dedup)
                    if url_exists(conn, url):
                        print(f"⏭️  Skipped (already in DB): {url}")
                        continue

                    sb.open(url)
                    detail_html = sb.get_page_source()
                    data = scrape_novel_with_bs4(detail_html, url)

                    if data:
                        upsert_novel(conn, data)
                        conn.commit()
                        print(f"✅ Saved: {data['Title']}")
                    else:
                        print(f"❌ Failed scrape: {url}")

                if max_pages and page_count >= max_pages:
                    print(f"Reached max_pages ({max_pages}). Stopping.")
                    break

                if not go_to_next_page(sb, page_count + 1):
                    print("No more pages. Scraping complete.")
                    break

    print(f"\n✅ All done! Data saved to {db_file}")

if __name__ == "__main__":
    main()
