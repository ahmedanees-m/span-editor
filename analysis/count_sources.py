"""Tabulate the source register by role, status and readout class."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from span_editor.io import read_corpus, read_sources

DEFAULT_GATE = 12


def summarise(sources, entries=(), gate: int = DEFAULT_GATE) -> dict:
    architecture = [s for s in sources if s.is_architecture_source]
    confirmed = [s for s in architecture if s.status == "confirmed"]
    quantitative = [s for s in confirmed if s.readout_class == "amplicon_sequencing"]

    curated_keys = {entry.source_key for entry in entries}
    curated = [s for s in architecture if s.key in curated_keys]
    blocked = {s.key: s.status for s in architecture if s.key not in curated_keys}

    return {
        "architecture_curated": len(curated),
        "curated_keys": [s.key for s in curated],
        "uncurated": blocked,
        "gate": gate,
        "total_sources": len(sources),
        "by_role": dict(Counter(s.role for s in sources)),
        "architecture_sources": len(architecture),
        "architecture_confirmed": len(confirmed),
        "architecture_to_retrieve": len(
            [s for s in architecture if s.status == "to_retrieve"]
        ),
        "architecture_verify_primary": len(
            [s for s in architecture if s.status == "verify_primary"]
        ),
        "quantitative_core": len(quantitative),
        "records_resolved": len([s for s in sources if s.has_resolved_record]),
        "architecture_by_readout": dict(Counter(s.readout_class for s in architecture)),
        "gate_met": len(curated) >= gate,
        "keys": {
            "architecture": [s.key for s in architecture],
            "quantitative_core": [s.key for s in quantitative],
        },
    }


def render(summary: dict) -> str:
    lines = [
        f"sources registered            {summary['total_sources']}",
        f"architecture sources          {summary['architecture_sources']} registered",
        f"  confirmed                   {summary['architecture_confirmed']}",
        f"  primary to verify           {summary['architecture_verify_primary']}",
        f"  still to retrieve           {summary['architecture_to_retrieve']}",
        f"  curated into the corpus     {summary['architecture_curated']}",
        f"quantitative core             {summary['quantitative_core']}",
        f"records resolved to a doi     {summary['records_resolved']} of {summary['total_sources']}",
        f"gate ({summary['gate']} architecture sources)  "
        f"{'met' if summary['gate_met'] else 'not met'}",
        "",
        "architecture sources by readout class",
    ]
    for readout, count in sorted(summary["architecture_by_readout"].items()):
        lines.append(f"  {readout:24s} {count}")
    if summary["uncurated"]:
        lines.append("")
        lines.append("registered but not curated")
        for key, status in sorted(summary["uncurated"].items()):
            lines.append(f"  {key:20s} {status}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, default=Path("data/sources.tsv"))
    parser.add_argument("--corpus", type=Path, default=Path("data/architectures.tsv"))
    parser.add_argument("--gate", type=int, default=DEFAULT_GATE)
    parser.add_argument("--out", type=Path, default=Path("results/source_count.json"))
    args = parser.parse_args(argv)

    entries = read_corpus(args.corpus) if args.corpus.exists() else []
    summary = summarise(read_sources(args.sources), entries, gate=args.gate)
    print(render(summary))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
