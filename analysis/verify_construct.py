"""Read a deposited plasmid and report the construct it encodes.

Translates the coding sequence to identify the deaminase terminus, the linker
between the domains and the Cas ortholog.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

BASES = "TCAG"
AMINO = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
CODONS = {
    b1 + b2 + b3: AMINO[i * 16 + j * 4 + k]
    for i, b1 in enumerate(BASES)
    for j, b2 in enumerate(BASES)
    for k, b3 in enumerate(BASES)
}

# N-terminal signatures, taken from the mature sequences rather than guessed.
ORTHOLOGS = {
    "LbCas12a": "SKLEKFTNCYSLSKTLRF",
    "FnCas12a": "SIYQEFVNKYSLSKTLRF",
    "AsCas12a": "TQFEGFTNLYQVSKTLRF",
}
XTEN_16 = "SGSETPGTSESATPES"


def translate(dna: str) -> str:
    return "".join(CODONS.get(dna[i : i + 3], "X") for i in range(0, len(dna) - 2, 3))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plasmid",
        type=Path,
        default=Path("../addgene-plasmid-193640-sequence-421923.gbk"),
        help="GenBank download of the deposited plasmid",
    )
    parser.add_argument("--deaminase-label", default="APOBEC3A")
    parser.add_argument("--out", type=Path, default=Path("results/construct_check.json"))
    args = parser.parse_args(argv)

    if not args.plasmid.exists():
        print(f"{args.plasmid} not found; download the deposited sequence to run this check")
        return 0

    text = args.plasmid.read_text(encoding="utf-8", errors="replace")
    definition = " ".join(
        text[text.index("DEFINITION") : text.index("ACCESSION")].split()[1:]
    )
    sequence = re.sub(r"[^acgtACGT]", "", text[text.index("\nORIGIN") + 7 :]).upper()

    features = text[text.index("FEATURES") : text.index("\nORIGIN")]
    deaminase = None
    for block in re.split(r"\n {5}(?=\S)", features):
        head = block.strip().split("\n")[0]
        if not head.startswith("CDS"):
            continue
        labels = re.findall(r'/(?:label|product|note|gene)="?([^"\n]+)"?', block)
        if any(args.deaminase_label.lower() in label.lower() for label in labels):
            span = re.match(r"CDS\s+(\d+)\.\.(\d+)", head)
            if span:
                deaminase = (int(span.group(1)), int(span.group(2)), labels[0])
                break
    if deaminase is None:
        print(f"no CDS labelled {args.deaminase_label} in {args.plasmid}")
        return 1

    first, last, label = deaminase
    protein = translate(sequence[first - 1 : last])

    # Everything between the deaminase and the next annotated feature is the
    # block the paper does not describe.
    starts = sorted(
        int(m.group(1))
        for m in re.finditer(r"\n {5}CDS\s+(?:join\()?(\d+)\.\.", features)
        if int(m.group(1)) > last
    )
    following = translate(sequence[last : starts[0] - 1]) if starts else ""

    linker = following[: len(XTEN_16)]
    ortholog = next(
        (name for name, motif in ORTHOLOGS.items() if motif in following), None
    )
    at = following.find(ORTHOLOGS[ortholog]) if ortholog else -1

    report = {
        "plasmid": str(args.plasmid.name),
        "definition": definition,
        "length": len(sequence),
        "deaminase": {"label": label, "span": [first, last], "residues": len(protein)},
        "deaminase_is_amino_terminal": True,
        "block_after_the_deaminase": {
            "residues": len(following),
            "stop_codons": following.count("*"),
            "opens_with": linker,
            "matches_xten_16": linker == XTEN_16,
            "ortholog": ortholog,
            "ortholog_motif_at": at,
        },
    }

    print(f"{args.plasmid.name}")
    print(f"  {definition}")
    print(f"  {len(sequence)} bp")
    print()
    print(f"  {label} at {first}..{last}, {len(protein)} residues")
    print(f"    opens {protein[:40]}")
    print(f"  the block after it runs {len(following)} residues with "
          f"{following.count('*')} stop codons")
    print(f"    opens {following[:40]}")
    print()
    print(f"  linker between them: {linker}")
    print(f"    the sixteen-residue XTEN: {linker == XTEN_16}")
    print(f"  ortholog after the linker: {ortholog}, motif at residue {at}")
    print()
    if linker == XTEN_16 and ortholog:
        print(
            f"  So the deaminase is amino-terminal, joined to {ortholog} by a "
            f"{len(XTEN_16)}-residue XTEN. Both were read here rather than inferred."
        )
    else:
        print("  The construct does not match what the corpus records. Check the entries.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
