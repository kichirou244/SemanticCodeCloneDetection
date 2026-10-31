#!/usr/bin/env python3
"""Script crawl du lieu ma nguon (source code) cac bai nop tu he thong CODER HUSC (coder.husc.edu.vn).

Cac tinh nang:
- Ho tro danh sach cac ma bai tap (problem IDs) can crawl (mac dinh: spit0019).
- Tu dong phan trang (pagination) de lay toan bo cac bai nop hop le (mac dinh: status=AC).
- Tai ma nguon goc (raw code) qua endpoint /src/{submission_id}/raw.
- Tu dong nhan dien duoi file dua tren ngon ngu (.cpp, .c, .py, .java, .cs, .pas,...).
- Ho tro co che bo qua (skip) neu file da ton tai, de dang tiep tuc khi bi gian doan.
- Luu tru day du metadata (nguoi nop, ngon ngu, thoi gian chay, bo nho, diem) ra file JSONL.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import re
import sys
import time
from urllib.parse import urljoin

from bs4 import BeautifulSoup
import requests

# ==============================================================================
# 1. THONG SO CAU HINH MAC DINH (USER CO THE CHINH SUA TRUC TIEP TAI DAY)
# ==============================================================================
DEFAULT_CSRFTOKEN = "3ERt9V2QJjWUgRn2Ye3zMG9hmUNIK6VM"
DEFAULT_SESSIONID = "hor0ywxdzkavlv72ehd6pt2nxpryt0ut"

# Danh sach cac bai tap can crawl (co the them nhieu ma bai vao day)
DEFAULT_PROBLEMS = [
    "spit0019", "spit0060", "spit0062", "spit0067", "spit0070", "spit0090", "spit0100", "spit0108"
]

DEFAULT_STATUS = "AC"  # Chi tai cac bai Accepted (AC). De trong "" neu muon lay tat ca
DEFAULT_OUT_DIR = "data/raw/coder"
DEFAULT_DELAY = 0.5  # Khoang nghi giua cac request (giay) de tranh bi chan / rate limit

BASE_URL = "https://coder.husc.edu.vn"

# ==============================================================================
# 2. LOGGING SETUP
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("coder_crawler")


# ==============================================================================
# 3. HELPER FUNCTIONS
# ==============================================================================
def get_file_extension(language: str) -> str:
    """Xac dinh duoi file phu hop voi ngon ngu lap trinh duoc nop."""
    lang_lower = language.lower()
    if "c++" in lang_lower or "cpp" in lang_lower or "clang++" in lang_lower:
        return ".cpp"
    elif "c1" in lang_lower or lang_lower == "c" or "clang" in lang_lower:
        return ".c"
    elif "py" in lang_lower:
        return ".py"
    elif "java" in lang_lower:
        return ".java"
    elif "pas" in lang_lower:
        return ".pas"
    elif "c#" in lang_lower or "monocs" in lang_lower:
        return ".cs"
    elif "go" in lang_lower:
        return ".go"
    elif "rust" in lang_lower:
        return ".rs"
    elif "kotlin" in lang_lower:
        return ".kt"
    elif "js" in lang_lower or "javascript" in lang_lower:
        return ".js"
    return ".cpp"  # Mac dinh la .cpp neu khong xac dinh ro


def clean_filename(name: str) -> str:
    """Loai bo ky tu dac biet khoi ten file de tranh loi he dieu hanh."""
    return re.sub(r'[\\/*?:"<>| ]', "_", name)


class CoderHuscCrawler:
    """Trinh thu thap du lieu ma nguon tu he thong CODER HUSC."""

    def __init__(
        self,
        csrftoken: str,
        sessionid: str,
        out_dir: str | Path = DEFAULT_OUT_DIR,
        delay: float = DEFAULT_DELAY,
        status: str = DEFAULT_STATUS,
    ):
        self.csrftoken = csrftoken
        self.sessionid = sessionid
        self.out_dir = Path(out_dir)
        self.delay = delay
        self.status = status

        self.session = requests.Session()
        self.session.cookies.update(
            {
                "csrftoken": self.csrftoken,
                "sessionid": self.sessionid,
            }
        )
        self.session.headers.update(
            {
                "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
                "accept-language": "vi,en-US;q=0.9,en;q=0.8",
                "cache-control": "max-age=0",
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/130.0.0.0 Safari/537.36"
                ),
            }
        )

    def fetch_submissions_page(self, url: str) -> tuple[list[dict], str | None]:
        """Lay danh sach cac bai nop tren mot trang submissions va tim link trang ke tiep."""
        time.sleep(self.delay)
        resp = self.session.get(url, timeout=15)
        if resp.status_code != 200:
            logger.error("Loi HTTP %s khi truy cap %s", resp.status_code, url)
            return [], None

        soup = BeautifulSoup(resp.text, "html.parser")
        submissions = []

        # Parse bang danh sach bai nop (fl-table hoac fl-subs-table)
        rows = soup.find_all("tr", class_="fl-subs-row")
        for row in rows:
            sub_id = row.get("id") or ""
            result = row.get("data-result") or ""

            # Loc theo status neu co yeu cau
            if self.status and result.upper() != self.status.upper():
                continue

            # Nguoi nop (user)
            user_el = row.find("a", class_="fl-user")
            username = user_el.get_text(strip=True) if user_el else "unknown"

            # Ngon ngu lap trinh (nam o cot index 3)
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

            # Thoi gian nop
            time_el = row.find("time")
            sub_time = time_el.get("datetime") or time_el.get_text(strip=True) if time_el else ""

            # Diem dat duoc
            points_el = row.find("span", class_="fl-subs-points")
            points = points_el.get_text(" ", strip=True) if points_el else ""

            submissions.append(
                {
                    "submission_id": sub_id,
                    "user": username,
                    "result": result,
                    "language": lang,
                    "runtime": runtime,
                    "memory": memory,
                    "time": sub_time,
                    "points": points,
                }
            )

        # Tim link trang ke tiep (Next Page)
        next_url = None
        # Tim nut "Trang sau"
        next_btn = soup.find("a", attrs={"aria-label": "Trang sau"})
        if next_btn and next_btn.get("href"):
            next_url = urljoin(url, next_btn["href"])
        else:
            # Thu kiem tra chevron_right trong fl-pager
            pager = soup.find("nav", class_="fl-pager")
            if pager:
                current_page_el = pager.find("a", attrs={"aria-current": "page"})
                if current_page_el:
                    next_sibling = current_page_el.find_next_sibling("a", class_="fl-pager__page")
                    if next_sibling and next_sibling.get("href"):
                        next_url = urljoin(url, next_sibling["href"])

        return submissions, next_url

    def fetch_raw_code(self, submission_id: str) -> str | None:
        """Tai ma nguon goc tu endpoint /src/{submission_id}/raw."""
        url = f"{BASE_URL}/src/{submission_id}/raw"
        headers = {
            "Referer": f"{BASE_URL}/submission/{submission_id}",
        }
        time.sleep(self.delay)
        try:
            resp = self.session.get(url, headers=headers, timeout=15)
            if resp.status_code == 200:
                return resp.text
            else:
                logger.warning("Khong the tai code cho #%s (HTTP %s)", submission_id, resp.status_code)
                return None
        except Exception as e:
            logger.error("Loi khi tai code cho #%s: %s", submission_id, e)
            return None

    def crawl_problem(
        self,
        problem_id: str,
        max_pages: int | None = None,
        max_submissions: int | None = None,
    ) -> dict:
        """Crawl toan bo bai nop cua mot bai tap."""
        logger.info("=" * 60)
        logger.info("BAT DAU CRAWL BAI TAP: %s (Status: %s)", problem_id, self.status or "ALL")
        logger.info("=" * 60)

        # Thu muc luu tru rieng cho bai tap
        problem_dir = self.out_dir / problem_id
        code_dir = problem_dir / "code"
        code_dir.mkdir(parents=True, exist_ok=True)

        meta_file = problem_dir / "metadata.jsonl"
        summary_file = problem_dir / "summary.json"

        # Doc metadata cu (neu co) de tranh crawl lai
        existing_sub_ids = set()
        if meta_file.exists():
            with open(meta_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            record = json.loads(line)
                            existing_sub_ids.add(record["submission_id"])
                        except Exception:
                            pass
            logger.info("Da tim thay %d bai nop da tai truoc do.", len(existing_sub_ids))

        # URL khoi dau
        start_url = f"{BASE_URL}/problem/{problem_id}/submissions/"
        if self.status:
            start_url += f"?status={self.status}"

        current_url = start_url
        page_idx = 1
        all_submissions: list[dict] = []

        # 1. Quet danh sach submissions qua tat ca cac trang
        while current_url:
            logger.info("[Trang %d] Dang lay danh sach tu: %s", page_idx, current_url)
            page_subs, next_url = self.fetch_submissions_page(current_url)

            if not page_subs:
                logger.info("Khong tim thay bai nop nao o trang %d. Dung quet trang.", page_idx)
                break

            logger.info("  -> Tim thay %d bai nop tai trang %d", len(page_subs), page_idx)
            all_submissions.extend(page_subs)

            if max_submissions and len(all_submissions) >= max_submissions:
                all_submissions = all_submissions[:max_submissions]
                logger.info("Dat gioi han toi da %d bai nop.", max_submissions)
                break

            if max_pages and page_idx >= max_pages:
                logger.info("Dat gioi han toi da %d trang.", max_pages)
                break

            if next_url == current_url:
                break
            current_url = next_url
            page_idx += 1

        logger.info("TONG CONG: Thu thap duoc %d bai nop cho bai %s.", len(all_submissions), problem_id)

        # 2. Tai ma nguon va ghi vao file
        downloaded_count = 0
        skipped_count = 0
        failed_count = 0

        with open(meta_file, "a", encoding="utf-8") as meta_out:
            for i, sub in enumerate(all_submissions, start=1):
                sub_id = sub["submission_id"]
                user = clean_filename(sub["user"])
                ext = get_file_extension(sub["language"])
                file_name = f"{sub_id}_{user}{ext}"
                file_path = code_dir / file_name

                # Kiem tra file da ton tai chua
                if file_path.exists() and file_path.stat().st_size > 0:
                    skipped_count += 1
                    continue

                logger.info("[%d/%d] Dang tai code #%s (User: %s, Lang: %s)...", i, len(all_submissions), sub_id, user, sub["language"])
                code_text = self.fetch_raw_code(sub_id)

                if code_text is not None:
                    file_path.write_text(code_text, encoding="utf-8")
                    sub["file_name"] = file_name
                    sub["file_path"] = str(file_path.relative_to(self.out_dir.parent))
                    sub["code_length"] = len(code_text)
                    sub["line_count"] = len(code_text.splitlines())

                    # Ghi ngay vao metadata.jsonl
                    meta_out.write(json.dumps(sub, ensure_ascii=False) + "\n")
                    meta_out.flush()
                    downloaded_count += 1
                else:
                    failed_count += 1

        summary = {
            "problem_id": problem_id,
            "status_filter": self.status,
            "total_submissions_found": len(all_submissions),
            "new_downloaded": downloaded_count,
            "already_existing": skipped_count,
            "failed": failed_count,
            "code_directory": str(code_dir),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        logger.info("HOAN TAT BAI %s:", problem_id)
        logger.info("  - Tai moi: %d", downloaded_count)
        logger.info("  - Da co tu truoc (skip): %d", skipped_count)
        logger.info("  - That bai: %d", failed_count)
        logger.info("  - Thu muc luu: %s", code_dir)
        return summary


# ==============================================================================
# 4. MAIN ENTRY POINT
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Script crawl code submissions tu he thong CODER HUSC (coder.husc.edu.vn)"
    )
    parser.add_argument(
        "--problems",
        "-p",
        nargs="+",
        default=DEFAULT_PROBLEMS,
        help="Danh sach cac ma bai tap can crawl (vd: spit0019 spit0020)",
    )
    parser.add_argument(
        "--csrftoken",
        type=str,
        default=DEFAULT_CSRFTOKEN,
        help="Gia tri cookie csrftoken",
    )
    parser.add_argument(
        "--sessionid",
        type=str,
        default=DEFAULT_SESSIONID,
        help="Gia tri cookie sessionid",
    )
    parser.add_argument(
        "--status",
        type=str,
        default=DEFAULT_STATUS,
        help="Ket qua loc (mac dinh: AC). De trong '' neu muon lay tat ca",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=DEFAULT_OUT_DIR,
        help="Thu muc goc luu tru code (mac dinh: data/raw/coder)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_DELAY,
        help="Khoang nghi giua cac request (giay, mac dinh: 0.5s)",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="So trang toi da can crawl cho moi bai tap (None = lay tat ca)",
    )
    parser.add_argument(
        "--max-submissions",
        type=int,
        default=None,
        help="So bai nop toi da can lay (None = lay tat ca)",
    )

    args = parser.parse_args()

    crawler = CoderHuscCrawler(
        csrftoken=args.csrftoken,
        sessionid=args.sessionid,
        out_dir=args.out_dir,
        delay=args.delay,
        status=args.status,
    )

    summaries = []
    for pid in args.problems:
        s = crawler.crawl_problem(
            problem_id=pid,
            max_pages=args.max_pages,
            max_submissions=args.max_submissions,
        )
        summaries.append(s)

    print("\n" + "=" * 60)
    print("TONG KET TOAN BO QUA TRINH CRAWL:")
    print("=" * 60)
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
