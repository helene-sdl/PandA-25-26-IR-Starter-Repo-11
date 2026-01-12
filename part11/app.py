#!/usr/bin/env python3
"""
Part 11 starter.

WHAT'S NEW IN PART 11. A positional Index. It's almost done, only the finishing touches remain.
"""
from typing import List
import time

from constants import BANNER, HELP
from models import SearchResult, Searcher

from file_utilities import load_config, load_sonnets, Configuration


def print_results(
    query: str | None,
    results: List[SearchResult],
    highlight_mode: str,
    query_time_ms: float | None = None,
    total_docs: int | None = None,
) -> None:
    if total_docs is None:
        total_docs = len(results)
    matched = [r for r in results if r.matches > 0]

    line = f'{len(matched)} out of {total_docs} sonnets contain "{query}".'
    if query_time_ms is not None:
        line += f" Your query took {query_time_ms:.2f}ms."
    print(line)

    for idx, r in enumerate(matched, start=1):
        r.print(idx, highlight_mode, total_docs)

# ---------- CLI loop ----------
def main() -> None:
    print(BANNER)
    config = load_config()

    # Load sonnets (from cache or API)
    start = time.perf_counter()
    sonnets = load_sonnets()

    elapsed = (time.perf_counter() - start) * 1000
    print(f"Loading sonnets took: {elapsed:.3f} [ms]")

    print(f"Loaded {len(sonnets)} sonnets.")

    searcher = Searcher(sonnets)

    while True:
        try:
            raw = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not raw:
            continue

        # commands
        if raw.startswith(":"):
            if raw == ":quit":
                print("Bye.")
                break

            if raw == ":help":
                print(HELP)
                continue

            if raw.startswith(":highlight"):
                parts = raw.split()
                if len(parts) == 2 and parts[1].lower() in ("on", "off"):
                    config.highlight = parts[1].lower() == "on"
                    print("Highlighting", "ON" if config.highlight else "OFF")
                    config.save()
                else:
                    print("Usage: :highlight on|off")
                continue

            if raw.startswith(":search-mode"):
                parts = raw.split()
                if len(parts) == 2 and parts[1].upper() in ("AND", "OR"):
                    config.search_mode = parts[1].upper()
                    print("Search mode set to", config.search_mode)
                    config.save()
                else:
                    print("Usage: :search-mode AND|OR")
                continue

            if raw.startswith(":hl-mode"):
                parts = raw.split()
                if len(parts) == 2 and parts[1].upper() in ("DEFAULT", "GREEN"):
                    config.hl_mode = parts[1].upper()
                    print("Highlight mode set to", config.hl_mode)
                    config.save()
                else:
                    print("Usage: :hl_mode DEFAULT|GREEN")
                continue

            continue

        # ---------- Query evaluation ----------

        words = raw.split()
        if not words:
            continue

        start = time.perf_counter()

        results = searcher.search(raw, config.search_mode)

        # Initialize elapsed_ms to contain the number of milliseconds the query evaluation took
        elapsed_ms = (time.perf_counter() - start) * 1000

        highlight_mode = config.hl_mode if config.highlight else None

        print_results(raw, results, highlight_mode, elapsed_ms, total_docs=searcher.total_sonnets) # added to fix the number of sonnets in output

if __name__ == "__main__":
    main()
