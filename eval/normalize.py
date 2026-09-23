"""Text normalization for evaluation matching.

Two rules that are easy to get wrong:
  1. Turkish: map İ to i and I to ı BEFORE lowercasing. Python's lower()
     turns İ into i plus a combining dot, so naive matching fails.
  2. Apostrophes are deleted, not replaced with a space, because the
     pipeline's clean_text() deletes them (KKB'nin becomes KKBnin).
"""

APOSTROPHES = ["'", "\u2019", "\u02bc", "`"]


def norm(s: str) -> str:
    s = s.replace("\u00ad", "")
    for ap in APOSTROPHES:
        s = s.replace(ap, "")
    s = s.replace("\u0130", "i").replace("I", "\u0131").lower()
    s = "".join(c if c.isalnum() or c.isspace() else " " for c in s)
    return " ".join(s.split())


def contains(haystack: str, needle: str) -> bool:
    return norm(needle) in norm(haystack)
