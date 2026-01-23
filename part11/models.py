from __future__ import annotations
from typing import List, Dict, Any, Tuple, Callable
import string
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

def normalize_token(token:str)->str:
    token = token.lower()
    token = token.replace("'", "").replace(",", "").replace(".", "")
    return token

def normalized_stem_token(token:str)->str:
    normalized_token = normalize_token(token)
    return stemmer.stem(normalized_token)


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

    def search_for(self: Sonnet, query: str, index: Index) -> SearchResult:
        stem = normalized_stem_token(query)

        if stem not in index.dictionary:
            return SearchResult(self.title, [], [], 0)

        postings = index.dictionary[stem].get(self.id, [])

        title_spans = []
        line_matches_dict = {}  # line_no -> spans

        for posting in postings:
            span = (posting.position, posting.position + len(posting.original_token))

            if posting.line_no is None:
                title_spans.append(span)
            else:
                line_matches_dict.setdefault(posting.line_no, []).append(span)

        line_matches = [
            LineMatch(ln, self.lines[ln - 1], spans)
            for ln, spans in sorted(line_matches_dict.items())
        ]

        total = len(title_spans) + sum(len(lm.spans) for lm in line_matches)

        return SearchResult(self.title, title_spans, line_matches, total)


class LineMatch:
    def __init__(self, line_no: int, text: str, spans: List[Tuple[int, int]]):
        self.line_no = line_no
        self.text = text
        self.spans = spans

    def copy(self):
        return LineMatch(self.line_no, self.text, self.spans)


class Posting:
    def __init__(self, line_no: int, position: int, original_token: str):
        self.line_no = line_no
        self.position = position
        self.original_token = original_token

    def __repr__(self) -> str:
        return f"{self.line_no}:{self.position}"


class Index:
    def __init__(self, sonnets: list[Sonnet]):
        self.sonnets = {sonnet.id: sonnet for sonnet in sonnets}
        self.dictionary = {}

        for sonnet in sonnets:
            for token, pos in self.tokenize(sonnet.title):
                stem = normalized_stem_token(token)
                if stem:
                    self._add_token(sonnet.id, stem, None, pos, token)

            for line_no, line in enumerate(sonnet.lines, start=1):
                for token, pos in self.tokenize(line):
                    stem = normalized_stem_token(token)
                    if stem:
                        self._add_token(sonnet.id, stem, line_no, pos, token)
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

    def _add_token(self, doc_id: int, stem: str, line_no: int | None, position: int, token: str):
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
        if stem not in self.dictionary:
            self.dictionary[stem] = {}

        postings = self.dictionary[stem]
        if doc_id not in postings:
            postings[doc_id] = []

        postings[doc_id].append(Posting(line_no, position, token))

    def search_for(self, token: str) -> dict[int, SearchResult]:
        # The dictionary results will have the id of the sonnet as its key and SearchResult as its value. You can
        # see its Type hint in the signature of the method.
        token = normalized_stem_token(token)
        results: dict[int, SearchResult] = {}

        if not token:
            return results

        if token in self.dictionary:
            postings_list = self.dictionary[token]
            for doc_id, postings in postings_list.items():
                sonnet = self.sonnets[doc_id]

                        # ToDo 3: Based on the posting create the corresponding SearchResult instance
                for posting in postings:
                    # Build SearchResult for each posting
                    if posting.line_no is None:
                        # title highlight
                        span = (posting.position, posting.position + len(posting.original_token))
                        sr = SearchResult(sonnet.title, [span], [], 1)
                    else:
                        line_text = sonnet.lines[posting.line_no - 1]
                        span = (posting.position, posting.position + len(posting.original_token))
                        lm = LineMatch(posting.line_no, line_text, [span])
                        sr = SearchResult(sonnet.title, [], [lm], 1)

                    if doc_id not in results:
                        results[doc_id] = sr
                    else:
                        results[doc_id] = results[doc_id].combine_with(sr)

        return results

class Searcher:
    def __init__(self, sonnets: List[Sonnet]):
        self.index = Index(sonnets)
        self.total_sonnets = len(sonnets)


    def search(self, query: str, search_mode: str) -> List[SearchResult]:

        words = query.split()

        combined_results = {}

        for word in words:
            # Searching for the word in all sonnets
            token = normalized_stem_token(word)
            results = self.index.search_for(token)

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
