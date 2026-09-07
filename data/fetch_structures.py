"""Download the structures listed in structures.txt from the RCSB.

Files land in data/structures as gzipped mmCIF. The directory is not tracked;
re-running the script is the way to reproduce it.
"""

from __future__ import annotations

import argparse
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

RCSB_TEMPLATE = "https://files.rcsb.org/download/{accession}.cif.gz"
ACCESSION = re.compile(r"^([0-9][A-Za-z0-9]{3})\b")


def read_accessions(listing: Path) -> list[str]:
    accessions = []
    for line in listing.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = ACCESSION.match(stripped)
        if match:
            accessions.append(match.group(1).upper())
    return accessions


def download(accession: str, destination: Path, overwrite: bool = False) -> Path:
    target = destination / f"{accession}.cif.gz"
    if target.exists() and not overwrite:
        return target
    url = RCSB_TEMPLATE.format(accession=accession)
    with urllib.request.urlopen(url, timeout=60) as response:
        payload = response.read()
    target.write_bytes(payload)
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--listing", type=Path, default=Path(__file__).with_name("structures.txt"))
    parser.add_argument("--out", type=Path, default=Path(__file__).with_name("structures"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    accessions = read_accessions(args.listing)
    if not accessions:
        print(f"no accessions found in {args.listing}", file=sys.stderr)
        return 1

    failed = []
    for accession in accessions:
        try:
            path = download(accession, args.out, overwrite=args.overwrite)
        except (urllib.error.URLError, TimeoutError) as error:
            failed.append((accession, error))
            print(f"{accession}: {error}", file=sys.stderr)
        else:
            print(f"{accession}: {path.name} ({path.stat().st_size} bytes)")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
