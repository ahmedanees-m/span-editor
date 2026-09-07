"""Compare buildable tether lengths against a TALE-based system.

The source varies both connectors together, so the comparison is qualitative.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

TALE_SCAFFOLDS = {"C40": 43, "C11": 20, "C0": 3}
TALE_SHORTEST_THAT_EDITS = 20


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inversion", type=Path, default=Path("results/inversion.json"))
    parser.add_argument("--out", type=Path, default=Path("results/tale_check.json"))
    args = parser.parse_args(argv)

    if not args.inversion.exists():
        print(f"{args.inversion} not found; run analysis/invert.py first")
        return 1
    grid = json.loads(args.inversion.read_text("utf-8"))["grid"]

    buildable = defaultdict(list)
    failed = defaultdict(list)
    for row in grid:
        key = (row["site"], row["composition"])
        (buildable if row["buildable"] else failed)[key].append(row["length"])

    shortest = {
        f"{site}/{composition}": min(lengths)
        for (site, composition), lengths in sorted(buildable.items())
    }
    unbuildable = {
        f"{site}/{composition}": sorted(lengths)
        for (site, composition), lengths in sorted(failed.items())
    }

    flexible = {key: value for key, value in shortest.items() if key.endswith("/xten")}

    print("shortest connector that places the domain at all, by attachment site")
    for key, value in sorted(shortest.items()):
        blocked = unbuildable.get(key, [])
        note = f", nothing shorter than this works of {blocked}" if blocked else ""
        print(f"  {key:28s} {value:3d} residues{note}")
    print()
    print("TALE scaffolds, Feola et al.")
    for name, length in sorted(TALE_SCAFFOLDS.items(), key=lambda item: -item[1]):
        works = length >= TALE_SHORTEST_THAT_EDITS
        print(f"  {name:4s} {length:3d} residues  {'edits' if works else 'does not edit'}")
    print()
    pairs = set(shortest) | set(unbuildable)
    never = sorted(key for key in unbuildable if key not in shortest)
    grid_shortest = min(
        length for lengths in list(buildable.values()) + list(failed.values()) for length in lengths
    )
    short_end = sorted(
        key
        for key, value in shortest.items()
        if value > grid_shortest and grid_shortest in unbuildable.get(key, [])
    )
    print(
        f"of {len(pairs)} site and composition pairs, {len(never)} place no conformation "
        f"outside the protein at any length on the grid, and {len(short_end)} fail at "
        f"{grid_shortest} residues while working at a longer one."
    )
    if never:
        print(f"  never buildable: {', '.join(never)}")
    if short_end:
        print(f"  buildable only above the shortest: {', '.join(short_end)}")
    print(
        "This agrees with the TALE result on one point only, that a connector of a few "
        "residues leaves the deaminase unable to reach its substrate. It is not evidence "
        "about the R-loop model, since the TALE system differs in both substrate rigidity "
        "and tether topology."
    )

    report = {
        "tale_scaffolds": TALE_SCAFFOLDS,
        "tale_shortest_that_edits": TALE_SHORTEST_THAT_EDITS,
        "shortest_buildable": shortest,
        "lengths_with_no_conformation": unbuildable,
        "shortest_buildable_flexible": flexible,
        "status": "qualitative, secondary, both connectors varied together in the source",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
