#!/usr/bin/env python3
"""Thu thap du lieu ma nguon tu he thong CODER HUSC (coder.husc.edu.vn)."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import re
import time
from urllib.parse import urljoin

from bs4 import BeautifulSoup
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("coder_crawler")

BASE_URL = "https://coder.husc.edu.vn"
DEFAULT_PROBLEMS = ["spit0019", "spit0060", "spit0062", "spit0067", "spit0070", "spit0090", "spit0100", "spit0108"]


def get_file_extension(language: str) -> str:
    lang = language.lower()
    if any(k in lang for k in ["c++", "cpp", "clang++"]):
        return ".cpp"
    if any(k in lang for k in ["c1", "clang"]) or lang == "c":
        return ".c"
    if "py" in lang:
        return ".py"
    if "java" in lang:
        return ".java"
    if "pas" in lang:
        return ".pas"
    if "c#" in lang or "monocs" in lang:
        return ".cs"
    return ".cpp"


def clean_filename(name: str) -> str:
    return re.sub(r'[\\/*?:"<>| ]', "_", name)


class CoderHuscCrawler:
    def __init__(
        self,
        csrftoken: str,
        sessionid: str,
        out_dir: str | Path = "data/raw/coder",
        delay: float = 0.5,
        status: str = "AC",
    ):
        self.out_dir = Path(out_dir)
        self.delay = delay
        self.status = status
        self.session = requests.Session()
        self.session.cookies.update({"csrftoken": csrftoken, "sessionid": sessionid})
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        })

    def fetch_submissions_page(self, url: str) -> tuple[list[dict], str | None]:
        time.sleep(self.delay)
        resp = self.session.get(url, timeout=15)
        if resp.status_code != 200:
            logger.error("HTTP %s tai %s", resp.status_code, url)
            return [], None

        soup = BeautifulSoup(resp.text, "html.parser")
        submissions = []
        for row in soup.find_all("tr", class_="fl-subs-row"):
            sub_id = row.get("id") or ""
            result = row.get("data-result") or ""
            if self.status and result.upper() != self.status.upper():
                continue

            user_el = row.find("a", class_="fl-user")
            username = user_el.get_text(strip=True) if user_el else "unknown"

            tds = row.find_all("td")
            lang = ""
            runtime = ""
            memory = ""
            if len(tds) >= 4:
                span = tds[3].find("span")
                lang = span.get("title") if (span and span.get("title")) else tds[3].get_text(strip=True)
            if len(tds) >= 5:
                runtime = tds[4].get_text(strip=True)
            if len(tds) >= 6:
                memory = tds[5].get_text(strip=True)

            time_el = row.find("time")
            sub_time = time_el.get("datetime") or time_el.get_text(strip=True) if time_el else ""
            points_el = row.find("span", class_="fl-subs-points")
            points = points_el.get_text(" ", strip=True) if points_el else ""

            submissions.append({
                "submission_id": sub_id,
                "user": username,
                "result": result,
                "language": lang,
                "runtime": runtime,
                "memory": memory,
                "time": sub_time,
                "points": points,
            })

        next_url = None
        next_btn = soup.find("a", attrs={"aria-label": "Trang sau"})
        if next_btn and next_btn.get("href"):
            next_url = urljoin(url, next_btn["href"])
        else:
            pager = soup.find("nav", class_="fl-pager")
            if pager:
                cur = pager.find("a", attrs={"aria-current": "page"})
                if cur:
                    sibling = cur.find_next_sibling("a", class_="fl-pager__page")
                    if sibling and sibling.get("href"):
                        next_url = urljoin(url, sibling["href"])

        return submissions, next_url

    def fetch_raw_code(self, submission_id: str) -> str | None:
        url = f"{BASE_URL}/src/{submission_id}/raw"
        time.sleep(self.delay)
        try:
            resp = self.session.get(url, headers={"Referer": f"{BASE_URL}/submission/{submission_id}"}, timeout=15)
            return resp.text if resp.status_code == 200 else None
        except Exception as e:
            logger.error("Loi tai #%s: %s", submission_id, e)
            return None

    def crawl_problem(self, problem_id: str, max_pages: int | None = None, max_submissions: int | None = None) -> dict:
        code_dir = self.out_dir / problem_id / "code"
        code_dir.mkdir(parents=True, exist_ok=True)
        meta_file = self.out_dir / problem_id / "metadata.jsonl"
        summary_file = self.out_dir / problem_id / "summary.json"

        existing = set()
        if meta_file.exists():
            with open(meta_file, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            existing.add(json.loads(line)["submission_id"])
                        except Exception:
                            pass

        start_url = f"{BASE_URL}/problem/{problem_id}/submissions/"
        if self.status:
            start_url += f"?status={self.status}"

        cur_url = start_url
        page_idx = 1
        all_subs = []

        while cur_url:
            page_subs, next_url = self.fetch_submissions_page(cur_url)
            if not page_subs:
                break
            all_subs.extend(page_subs)
            if max_submissions and len(all_subs) >= max_submissions:
                all_subs = all_subs[:max_submissions]
                break
            if max_pages and page_idx >= max_pages:
                break
            if next_url == cur_url:
                break
            cur_url = next_url
            page_idx += 1

        downloaded, skipped, failed = 0, 0, 0
        with open(meta_file, "a", encoding="utf-8") as meta_out:
            for sub in all_subs:
                sub_id = sub["submission_id"]
                user = clean_filename(sub["user"])
                ext = get_file_extension(sub["language"])
                file_name = f"{sub_id}_{user}{ext}"
                file_path = code_dir / file_name

                if file_path.exists() and file_path.stat().st_size > 0:
                    skipped += 1
                    continue

                code = self.fetch_raw_code(sub_id)
                if code is not None:
                    file_path.write_text(code, encoding="utf-8")
                    sub.update({
                        "file_name": file_name,
                        "file_path": str(file_path.relative_to(self.out_dir.parent)),
                        "code_length": len(code),
                        "line_count": len(code.splitlines()),
                    })
                    meta_out.write(json.dumps(sub, ensure_ascii=False) + "\n")
                    meta_out.flush()
                    downloaded += 1
                else:
                    failed += 1

        summary = {
            "problem_id": problem_id,
            "status_filter": self.status,
            "total_found": len(all_subs),
            "downloaded": downloaded,
            "skipped": skipped,
            "failed": failed,
            "code_dir": str(code_dir),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        return summary


def main():
    parser = argparse.ArgumentParser(description="Thu thap code tu CODER HUSC")
    parser.add_argument("--problems", "-p", nargs="+", default=DEFAULT_PROBLEMS)
    parser.add_argument("--csrftoken", default=os.getenv("CODER_CSRFTOKEN", ""))
    parser.add_argument("--sessionid", default=os.getenv("CODER_SESSIONID", ""))
    parser.add_argument("--status", default="AC")
    parser.add_argument("--out-dir", default="data/raw/coder")
    parser.add_argument("--delay", type=float, default=0.5)
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--max-submissions", type=int, default=None)
    args = parser.parse_args()

    crawler = CoderHuscCrawler(
        csrftoken=args.csrftoken,
        sessionid=args.sessionid,
        out_dir=args.out_dir,
        delay=args.delay,
        status=args.status,
    )
    for pid in args.problems:
        crawler.crawl_problem(pid, max_pages=args.max_pages, max_submissions=args.max_submissions)


if __name__ == "__main__":
    main()
