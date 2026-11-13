"""Verify that the local news file matches the pinned upstream corpus prefix."""
import hashlib
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOCAL = ROOT / "data" / "raw" / "kurdish_news_corpus_source.txt"
URL = ("https://huggingface.co/datasets/abdulhade/KurdishTextCorpus/resolve/"
       "2675e164dda260088c729fded823e34b169f7424/ckb%20text%20corpus.txt")
PREFIX_BYTES = 1_250_000
EXPECTED = "7e21d6a49c11a263e7fcce5eb83c8737bc6783563cc4fc0589dbadb8cd55aef3"


def main():
    request = urllib.request.Request(
        URL, headers={"User-Agent": "KurdRAG-provenance-check", "Range": f"bytes=0-{PREFIX_BYTES - 1}"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        prefix = response.read(PREFIX_BYTES)
    if len(prefix) != PREFIX_BYTES:
        raise ValueError(f"Expected {PREFIX_BYTES} upstream bytes, received {len(prefix)}")
    newline = prefix.rfind(b"\n")
    if newline < 0:
        raise ValueError("Upstream prefix contains no line ending")
    trimmed = prefix[:newline + 1]
    local = LOCAL.read_bytes()
    if hashlib.sha256(local).hexdigest() != EXPECTED:
        raise ValueError("Unexpected local news file hash")
    if trimmed != local:
        raise ValueError("Local news file does not match the trimmed upstream prefix")
    print(f"Verified exact match: {len(local)} bytes, SHA-256 {EXPECTED}")


if __name__ == "__main__":
    main()
