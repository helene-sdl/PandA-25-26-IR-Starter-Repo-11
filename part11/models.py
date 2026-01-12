from __future__ import annotations
from typing import List, Dict, Any, Tuple, Callable
import string

class Sonnet:
    def __init__(self, sonnet_data: Dict[str, Any]):
        self.title = sonnet_data["title"]
        self.lines = sonnet_data["lines"]

        # ToDo 1: Make sure the sonnet has an attribute id that contains the number of the Sonnet as an int
        self.id = int(self.title.split()[1].rstrip(":"))

    @staticmethod
    def find_spans(text: str, pattern: str):
        """Return [(start, end), ...] for all (possibly overlapping) matches.
        Inputs should already be lowercased by the caller."""
        spans = []
        if not pattern:
            return spans

        for i in range(len(text) - len(pattern) + 1):
            if text[i:i + len(pattern)] == pattern:
                spans.append((i, i + len(pattern)))
        return spans

    def search_for(self: Sonnet, query: str) -> SearchResult:
        title_raw = str(self.title)
        lines_raw = self.lines

        q = query.lower()
        title_spans = self.find_spans(title_raw.lower(), q)

        line_matches = []
        for idx, line_raw in enumerate(lines_raw, start=1):  # 1-based line numbers
            spans = self.find_spans(line_raw.lower(), q)
            if spans:
                line_matches.append(LineMatch(idx, line_raw, spans))

        total = len(title_spans) + sum(len(lm.spans) for lm in line_matches)

        return SearchResult(title_raw, title_spans, line_matches, total)


class LineMatch:
    def __init__(self, line_no: int, text: str, spans: List[Tuple[int, int]]):
        self.line_no = line_no
        self.text = text
        self.spans = spans

    def copy(self):
        return LineMatch(self.line_no, self.text, self.spans)


class Posting:
    def __init__(self, line_no: int, position: int):
        self.line_no = line_no
        self.position = position

    def __repr__(self) -> str:
        return f"{self.line_no}:{self.position}"


class Index:
    def __init__(self, sonnets: list[Sonnet]):
        self.sonnets = {sonnet.id: sonnet for sonnet in sonnets}
        self.dictionary = {}

        for sonnet in sonnets:
            for token, pos in self.tokenize(sonnet.title):
                token = token.lower().strip(string.punctuation)
                if token:
                    self._add_token(sonnet.id, token, None, pos)

            for line_no, line in enumerate(sonnet.lines, start=1):
                for token, pos in self.tokenize(line):
                    token = token.lower().strip(string.punctuation)
                    if token:
                        self._add_token(sonnet.id, token, line_no, pos)
                # ToDo 2: Implement logic of adding tokens to the index. Use the pre-defined methods tokenize and
            #  _add_token to do so. Index the title and the lines of the sonnet.

    @staticmethod
    def tokenize(text):
        """
         Split a text string into whitespace-separated tokens.

         Each token is returned together with its starting character
         position in the input string.

         Args:
             text: The input text to tokenize.

         Returns:
             A list of (token, position) tuples, where:
             - token is a non-whitespace substring of `text`
             - position is the 0-based start index of the token in `text`
         """
        import re
        tokens = [
            (match.group(), match.start())
            for match in re.finditer(r"\S+", text)
        ]

        return tokens

    def _add_token(self, doc_id: int, token: str, line_no: int | None, position: int):
        """
        Add a single token occurrence to the inverted index.

        This method updates `self.dictionary`, which maps each token to a postings
        list. A postings list is a mapping from document ID to a list of `Posting`
        objects describing every occurrence of that token in the document.

        The resulting structure has the form:
            self.dictionary[token][doc_id] -> [Posting(line_no, position), ...]

        Where:
          - `line_no` identifies *where* in the document the token appears:
              * `None` means the token came from the title.
              * `0..n-1` means the token came from the corresponding line in
                `sonnet.lines`.
          - `position` is the 0-based character offset of the token within the
            corresponding text:
              * If `line_no is None`, it is the character offset within the full
                title string (including any prefix before ": "), as calculated by
                the caller.
              * Otherwise, it is the character offset within that line string.

        This method does not normalize tokens (e.g., lowercasing, punctuation
        stripping) and does not deduplicate occurrences; every call appends a new
        `Posting`.

        Args:
            doc_id: The ID of the document (sonnet) the token belongs to.
            token: The token text to index (as produced by `tokenize`).
            line_no: The line number within the document, or `None` for title tokens.
            position: The 0-based starting character index of the token within the
                title (if `line_no is None`) or within the line (otherwise).
        """
        if token not in self.dictionary:
            self.dictionary[token] = {}

        postings_list = self.dictionary[token]

        if doc_id not in postings_list:
            postings_list[doc_id] = []
        postings_list[doc_id].append(Posting(line_no, position))

    def search_for(self, token: str) -> dict[int, SearchResult]:
        """
        Search the index for a single token and return all matching documents.

        This method looks up the given token in the inverted index
        (`self.dictionary`). For every occurrence of the token in every indexed
        document, it constructs a `SearchResult` describing where the token
        appears and aggregates all occurrences per document.

        Each occurrence (posting) contributes:
          - a title match if `posting.line_no is None`, or
          - a line match if the token occurs in one of the sonnet’s lines.

        Multiple occurrences within the same document are merged using
        `SearchResult.combine_with`, so the final result contains all title
        spans, line matches, and an accumulated score for that document.

        Args:
            token: The token to search for. The token must match exactly how it
                was indexed (no normalization is performed).

        Returns:
            A dictionary mapping sonnet IDs to `SearchResult` objects.

            For each entry:
              - the key is the document (sonnet) ID
              - the value is a `SearchResult` containing:
                  * the sonnet title,
                  * all matching spans in the title,
                  * all matching lines with their spans,
                  * a score reflecting the total number of occurrences of `token`
                    in that document

            If the token does not exist in the index, an empty dictionary is
            returned.

        Notes:
            - This method performs an exact-token lookup; it does not support
              stemming, case folding, or partial matches.
            - Each posting contributes a score of 1 before aggregation.
        """
        # The dictionary results will have the id of the sonnet as its key and SearchResult as its value. You can
        # see its Type hint in the signature of the method.
        results = {}

        if token in self.dictionary:
            postings_list = self.dictionary[token]
            for doc_id, postings in postings_list.items():
                for posting in postings:
                    sonnet = self.sonnets[doc_id]

                    # ToDo 3: Based on the posting create the corresponding SearchResult instance
                    if posting.line_no is None:
                        span = (posting.position, posting.position + len(token))
                        result = SearchResult(sonnet.title, [span], [], 1)
                    else:
                        line_text = sonnet.lines[posting.line_no - 1]
                        span = (posting.position, posting.position + len(token))
                        lm = LineMatch(posting.line_no, line_text, [span])
                        result = SearchResult(sonnet.title, [], [lm], 1)
                    # At this point result contains the SearchResult corresponding to the posting - ready to be added
                    # to the results dictionary.
                    if doc_id not in results:
                        results[doc_id] = result
                    else:
                        results[doc_id] = results[doc_id].combine_with(result)

        return results

class Searcher:
    def __init__(self, sonnets: List[Sonnet]):
        self.index = Index(sonnets)
        self.total_sonnets = len(sonnets)


    def search(self, query: str, search_mode: str) -> List[SearchResult]:
        """
        Search sonnets for a multi-word query and return combined matches.

        The query is split on whitespace into individual words. Each word is looked
        up independently via `Index.search_for`, producing a dictionary:

            {sonnet_id: SearchResult}

        These per-word results are then merged across words according to the chosen
        `search_mode`:

          - "AND": Only sonnets that appear in the results of *every* query word
            are kept. For those sonnets, the corresponding `SearchResult` objects
            are merged using `SearchResult.combine_with`.

          - "OR": Sonnets that appear in the results of *any* query word are kept.
            If a sonnet matches multiple words, their `SearchResult` objects are
            merged using `SearchResult.combine_with`. If it matches only some
            words, its existing `SearchResult` is included as-is.

        In both modes, whenever a sonnet is present in both the accumulated results
        and the current word’s results, the two `SearchResult` objects are always
        merged so that all spans/line matches and scores are aggregated.

        Args:
            query: A whitespace-separated query string. Each word is searched as an
                exact token (no normalization is performed).
            search_mode: Either "AND" or "OR", controlling whether documents must
                match all query words ("AND") or any query word ("OR").

        Returns:
            A list of combined `SearchResult` objects, one per matching sonnet,
            sorted alphabetically by `SearchResult.title`.

        Notes:
            - This is an exact-token search: case differences, punctuation, and
              other normalization concerns must be handled before calling `search`
              if desired.
            - The final score in each `SearchResult` depends on how
              `SearchResult.combine_with` aggregates scores across occurrences and
              across query words.
        """
        words = query.split()

        combined_results = {}

        for word in words:
            # Searching for the word in all sonnets
            word = word.lower()
            results = self.index.search_for(word)

            # ToDo 4: Combine the search results from the search_for method of the index. From ToDo 2 you know
            #         that results is a dictionary with the key-value pairs of int-SearchResult, where the key is the
            #         document ID (the sonnet ID) and the value is the SearchResult for the current word in this sonnet.
            #         Re-think the combine logic. You need to check the keys of combined_results and results to find
            #         out whether both contain search results for certain sonnets. If both contains results, you will
            #         need to merge them independent of whether the current search mode is "AND" or "OR". But the "OR"
            #         mode will always contains all search results.

            # Add your code here...
            if not combined_results:
                combined_results = results.copy()
                continue

            if search_mode == "OR":
                for doc_id, r in results.items():
                    if doc_id in combined_results:
                        combined_results[doc_id] = combined_results[doc_id].combine_with(r)
                    else:
                        combined_results[doc_id] = r
            else:
                deleted = []
                for doc_id in combined_results:
                    if doc_id in results:
                        combined_results[doc_id] = combined_results[doc_id].combine_with(results[doc_id])
                    else:
                        deleted.append(doc_id)
                for d in deleted:
                    del combined_results[d]
            # At this point combined_results contains a dictionary with the sonnet ID as key and the search result for
            # this sonnet. Just like the result you receive from the index, but combined for all words

        results = list(combined_results.values())
        return sorted(results, key=lambda sr: sr.title)


class SearchResult:
    def __init__(self, title: str, title_spans: List[Tuple[int, int]], line_matches: List[LineMatch],
                 matches: int) -> None:
        self.title = title
        self.title_spans = title_spans
        self.line_matches = line_matches
        self.matches = matches

    def copy(self):
        return SearchResult(self.title, self.title_spans, self.line_matches, self.matches)

    @staticmethod
    def ansi_highlight(text: str, spans, highlight_mode) -> str:
        """Return text with ANSI highlight escape codes inserted."""
        if not spans:
            return text

        spans = sorted(spans)
        merged = []

        # Merge overlapping spans
        current_start, current_end = spans[0]
        for s, e in spans[1:]:
            if s <= current_end:
                current_end = max(current_end, e)
            else:
                merged.append((current_start, current_end))
                current_start, current_end = s, e
        merged.append((current_start, current_end))

        ansi_sequence = "\033[43m\033[30m" if highlight_mode == "DEFAULT" else "\033[1;92m"

        # Build highlighted string
        out = []
        i = 0
        for s, e in merged:
            out.append(text[i:s])
            out.append(ansi_sequence)  # yellow background, black text
            out.append(text[s:e])
            out.append("\033[0m")  # reset
            i = e
        out.append(text[i:])
        return "".join(out)

    def print(self, idx, highlight_mode: str | None, total_docs):
        title_line = (
            self.ansi_highlight(self.title, self.title_spans, highlight_mode)
            if highlight_mode
            else self.title
        )
        print(f"\n[{idx}/{total_docs}] {title_line}")
        for lm in self.line_matches:
            line_out = (
                self.ansi_highlight(lm.text, lm.spans, highlight_mode)
                if highlight_mode
                else lm.text
            )
            print(f"  [{lm.line_no:2}] {line_out}")

    def combine_with(self: SearchResult, other: SearchResult) -> SearchResult:
        """Combine two search results."""

        combined = self.copy()  # shallow copy

        combined.matches = self.matches + other.matches
        combined.title_spans = sorted(self.title_spans + other.title_spans)

        # Merge line_matches by line number
        lines_by_no = {lm.line_no: lm.copy() for lm in self.line_matches}
        for lm in other.line_matches:
            ln = lm.line_no
            if ln in lines_by_no:
                # extend spans & keep original text
                lines_by_no[ln].spans.extend(lm.spans)
            else:
                lines_by_no[ln] = lm.copy()

        combined.line_matches = sorted(lines_by_no.values(), key=lambda lm: lm.line_no)

        return combined
