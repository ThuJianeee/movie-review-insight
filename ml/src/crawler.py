"""Web crawler (assignment requirement 'c').

Crawls user reviews from IMDb title pages and saves them to data/crawled_reviews.csv.
IMDb star ratings are mapped to sentiment labels (>=7 positive, <=4 negative,
5-6 dropped as neutral) so the crawled data can be used exactly like the
Cornell/IMDb benchmark datasets.

Usage:
    python crawler.py --titles tt0111161 tt0068646 --pages 3
    python crawler.py --titles tt0111161 --max_reviews 200
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import re
import time

import requests
from bs4 import BeautifulSoup

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}
BASE = "https://www.imdb.com/title/{tconst}/reviews/_ajax"


def _label_from_rating(rating: int | None) -> int | None:
    if rating is None:
        return None
    if rating >= 7:
        return 1
    if rating <= 4:
        return 0
    return None  # neutral -> skipped for binary classification


def _parse_page(html: str) -> tuple[list[dict], str | None]:
    soup = BeautifulSoup(html, "html.parser")
    rows: list[dict] = []
    for block in soup.select("div.review-container, article.user-review-item"):
        body = block.select_one("div.text, div.ipc-html-content-inner-div")
        if body is None:
            continue
        text = body.get_text(" ", strip=True)
        rating_tag = block.select_one("span.rating-other-user-rating span, span.ipc-rating-star--rating")
        rating = None
        if rating_tag:
            m = re.search(r"\d+", rating_tag.get_text())
            rating = int(m.group()) if m else None
        label = _label_from_rating(rating)
        if label is None or len(text) < 80:
            continue
        rows.append({"text": text, "label": label, "rating": rating})

    key_tag = soup.select_one("div.load-more-data")
    next_key = key_tag.get("data-key") if key_tag else None
    return rows, next_key


def crawl_title(tconst: str, pages: int, delay: float) -> list[dict]:
    collected: list[dict] = []
    params = {"ref_": "undefined", "paginationKey": ""}
    for page in range(pages):
        url = BASE.format(tconst=tconst)
        try:
            resp = requests.get(url, headers=HEADERS, params=params, timeout=20)
            resp.raise_for_status()
        except requests.RequestException as exc:
            print(f"  request failed ({exc}) - stopping this title")
            break
        rows, next_key = _parse_page(resp.text)
        collected.extend(rows)
        print(f"  {tconst} page {page + 1}: +{len(rows)} usable reviews (total {len(collected)})")
        if not next_key:
            break
        params["paginationKey"] = next_key
        time.sleep(delay + random.uniform(0, 0.5))  # politeness / rate limiting
    return collected


def main(titles: list[str], pages: int, max_reviews: int | None, delay: float, out: str) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    all_rows: list[dict] = []
    for tconst in titles:
        print(f"Crawling {tconst} ...")
        all_rows.extend(crawl_title(tconst, pages, delay))
        if max_reviews and len(all_rows) >= max_reviews:
            all_rows = all_rows[:max_reviews]
            break

    path = os.path.join(DATA_DIR, out)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["text", "label", "rating"])
        writer.writeheader()
        writer.writerows(all_rows)

    pos = sum(r["label"] == 1 for r in all_rows)
    print(f"\nSaved {len(all_rows)} reviews ({pos} positive / {len(all_rows) - pos} negative) to {path}")
    print("Use it with:  python model1_naive_bayes.py --dataset crawled")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--titles", nargs="+",
                        default=["tt0111161", "tt0068646", "tt0468569", "tt1375666", "tt0816692"],
                        help="IMDb title IDs to crawl")
    parser.add_argument("--pages", type=int, default=3, help="Pages per title (25 reviews/page)")
    parser.add_argument("--max_reviews", type=int, default=None)
    parser.add_argument("--delay", type=float, default=1.5, help="Seconds between requests")
    parser.add_argument("--out", default="crawled_reviews.csv")
    args = parser.parse_args()
    main(args.titles, args.pages, args.max_reviews, args.delay, args.out)
