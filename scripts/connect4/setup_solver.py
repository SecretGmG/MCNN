"""Install the pinned Pascal Pons solver and 7x6 opening book locally."""

from __future__ import annotations

import argparse
import subprocess
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TARGET = ROOT / "tools/connect4-solver"
REPOSITORY = "https://github.com/PascalPons/connect4.git"
REVISION = "d6ba50d8aaf2308c769d9bf2abd42d90f34baf41"
BOOK_URL = "https://github.com/PascalPons/connect4/releases/download/book/7x6.book"


def install(target: Path) -> None:
    if not (target / ".git").is_dir():
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", REPOSITORY, str(target)], check=True
        )
    subprocess.run(["git", "checkout", REVISION], cwd=target, check=True)
    subprocess.run(["make"], cwd=target, check=True)
    book = target / "7x6.book"
    if not book.is_file():
        temporary = target / "7x6.book.download"
        urllib.request.urlretrieve(BOOK_URL, temporary)
        temporary.replace(book)
    print(f"solver={target / 'c4solver'}")
    print(f"opening_book={book}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    args = parser.parse_args()
    install(args.target)


if __name__ == "__main__":
    main()
