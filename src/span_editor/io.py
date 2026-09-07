"""Reading the corpus, the configuration files and the fit record.

Two tables carry the data. sources.tsv is the register of publications counted
in step 0; architectures.tsv is the corpus itself, one row per architecture and
readout. Observed profiles are stored inline as index:value pairs so that a row
stays legible in a diff.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from .geometry import NUMBERING_FROM_PAM_DISTAL
from .occupancy import ModelParameters, MotifPreference, Profile
from .tether import CompositionClass, EffectorSpec, LinkerSpec

__all__ = [
    "AssignedInputs",
    "CorpusEntry",
    "SourceRecord",
    "load_assigned_inputs",
    "load_parameters",
    "parse_profile",
    "parse_window",
    "read_corpus",
    "read_sources",
    "write_fit_record",
]

LABELS = {"tether_varied", "effector_varied", "scaffold_perturbed", "both", "none"}
READOUT_CLASSES = {
    "amplicon_sequencing",
    "functional_selection",
    "plant_system",
    "in_vitro_deamination",
    "sanger_deconvolution",
}
SPLITS = {"fit", "held_out"}


@dataclass(frozen=True)
class SourceRecord:
    """One publication in the step 0 register."""

    key: str
    citation: str
    role: str
    tether_variation: str
    readout_class: str
    status: str
    doi: str = ""
    notes: str = ""

    @property
    def is_architecture_source(self) -> bool:
        return self.role == "architecture"

    @property
    def is_quantitative(self) -> bool:
        return self.readout_class == "amplicon_sequencing"

    @property
    def has_resolved_record(self) -> bool:
        return bool(self.doi)


@dataclass
class CorpusEntry:
    """One architecture as measured in one publication."""

    entry_id: str
    source_key: str
    editor: str
    ortholog: str
    anchor_site: str
    linker_residues: int
    linker_composition: str
    effector: str
    label: str
    readout_class: str
    tier: str
    spacer_length: int
    numbering: str = NUMBERING_FROM_PAM_DISTAL
    structure_id: str = ""
    scaffold_variant: str = ""
    return_site: str = ""
    return_residues: int = 0
    split: str = "held_out"
    context: str = ""
    panel: str = ""
    observed: Profile | None = None
    observed_window: tuple[int, int] | None = None
    observed_width: int = 0

    def __post_init__(self) -> None:
        if self.label not in LABELS:
            raise ValueError(f"{self.entry_id}: unknown label {self.label!r}")
        if self.readout_class not in READOUT_CLASSES:
            raise ValueError(f"{self.entry_id}: unknown readout class {self.readout_class!r}")
        if self.split not in SPLITS:
            raise ValueError(f"{self.entry_id}: unknown split {self.split!r}")

    @property
    def linker(self) -> LinkerSpec:
        return LinkerSpec(n_residues=self.linker_residues, composition=self.linker_composition)

    @property
    def is_insertion(self) -> bool:
        """Whether the effector is inserted into the editor rather than fused to a terminus."""
        return bool(self.return_site and self.return_residues > 0)

    @property
    def fits_parameters(self) -> bool:
        """Whether the entry may contribute to fitting.

        Only sequencing readouts in the fit split qualify. Functional selections
        and plant systems are evaluated but never fitted, and never contribute
        to a width claim.
        """
        return self.split == "fit" and self.readout_class == "amplicon_sequencing"


@dataclass
class AssignedInputs:
    """The choices of section 7 that are set by judgment rather than fitted."""

    compositions: dict[str, CompositionClass]
    effectors: dict[str, EffectorSpec]
    motifs: dict[str, MotifPreference]
    coarse_radii: dict[str, float]
    structures: dict[str, str]
    scaffold_variants: dict[str, dict] = field(default_factory=dict)
    structure_geometry: dict[str, dict] = field(default_factory=dict)
    substrate_rise: float = 6.3
    observed_spread: float = 0.0
    surface_distance: float = 0.0
    occupancy_resolution: str = "residue"
    sampling: dict[str, int] = field(default_factory=dict)


def parse_profile(text: str) -> Profile | None:
    """Parse an inline profile written as index:value pairs separated by semicolons."""
    if not text or text.strip() in {"", "-", "pending"}:
        return None
    indices: list[int] = []
    values: list[float] = []
    for pair in text.split(";"):
        pair = pair.strip()
        if not pair:
            continue
        index, value = pair.split(":")
        indices.append(int(index))
        values.append(float(value))
    order = np.argsort(indices)
    return Profile(
        indices=np.asarray(indices, dtype=int)[order],
        values=np.asarray(values, dtype=float)[order],
    )


def parse_window(text: str) -> tuple[int, int] | None:
    """Parse a reported editing window written as first-last.

    Some sources give a per-position profile and some give only the span of
    positions they call the window. The second is worth carrying: it is enough
    to test whether a predicted window sits where the reported one does, which
    is the whole of what an architecture with no released table can settle.
    """
    if not text or text.strip() in {"", "-", "pending"}:
        return None
    first, _, last = text.strip().partition("-")
    return (int(first), int(last))


def format_profile(profile: Profile) -> str:
    return ";".join(
        f"{int(index)}:{value:g}" for index, value in zip(profile.indices, profile.values)
    )


def _read_tsv(path: Path | str) -> list[dict[str, str]]:
    """Read a tab-separated table, ignoring comment lines wherever they appear."""
    with open(path, newline="", encoding="utf-8") as handle:
        lines = [line for line in handle if not line.lstrip().startswith("#")]
    identifier = ("key", "entry_id")
    return [
        row
        for row in csv.DictReader(lines, delimiter="\t")
        if any((row.get(name) or "").strip() for name in identifier)
    ]


def read_sources(path: Path | str) -> list[SourceRecord]:
    return [
        SourceRecord(
            key=row["key"].strip(),
            citation=row["citation"].strip(),
            role=row["role"].strip(),
            tether_variation=row["tether_variation"].strip(),
            readout_class=row["readout_class"].strip(),
            status=row["status"].strip(),
            doi=row.get("doi", "").strip(),
            notes=row.get("notes", "").strip(),
        )
        for row in _read_tsv(path)
    ]


def read_corpus(path: Path | str) -> list[CorpusEntry]:
    entries = []
    for row in _read_tsv(path):
        entries.append(
            CorpusEntry(
                entry_id=row["entry_id"].strip(),
                source_key=row["source_key"].strip(),
                editor=row["editor"].strip(),
                ortholog=row["ortholog"].strip(),
                anchor_site=row["anchor_site"].strip(),
                linker_residues=int(row["linker_residues"]),
                linker_composition=row["linker_composition"].strip(),
                effector=row["effector"].strip(),
                label=row["label"].strip(),
                readout_class=row["readout_class"].strip(),
                tier=row["tier"].strip(),
                spacer_length=int(row["spacer_length"]),
                numbering=row.get("numbering", NUMBERING_FROM_PAM_DISTAL).strip(),
                structure_id=row.get("structure_id", "").strip(),
                scaffold_variant=row.get("scaffold_variant", "").strip(),
                return_site=row.get("return_site", "").strip(),
                return_residues=int(row.get("return_residues") or 0),
                split=row.get("split", "held_out").strip(),
                context=row.get("context", "").strip(),
                panel=row.get("panel", "").strip(),
                observed=parse_profile(row.get("observed_profile", "")),
                observed_window=parse_window(row.get("observed_window", "")),
                observed_width=int(row.get("observed_width") or 0),
            )
        )
    return entries


def load_parameters(path: Path | str) -> ModelParameters:
    """Read the fitted parameters.

    configs/params.yaml holds the search bounds and conventions and is written by
    hand; configs/fitted.yaml holds the fitted values and is written by the fit.
    The fitted file is preferred when it is present.
    """
    path = Path(path)
    fitted = path.with_name("fitted.yaml")
    source = fitted if fitted.exists() else path
    with open(source, encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    return ModelParameters.from_dict(document["fitted"])


def load_assigned_inputs(path: Path | str) -> AssignedInputs:
    with open(path, encoding="utf-8") as handle:
        document = yaml.safe_load(handle)

    compositions = {
        name: CompositionClass(
            name=name,
            persistence=float(values["persistence"]),
            rise=float(values["rise"]),
            departure_cone=float(values.get("departure_cone", 180.0)),
            rigid=bool(values.get("rigid", False)),
            persistence_range=(
                tuple(float(v) for v in values["range"]) if values.get("range") else None
            ),
            reference=values.get("reference", ""),
        )
        for name, values in document["linker_compositions"].items()
    }
    effectors = {
        name: EffectorSpec(
            name=name,
            active_site_offset=np.asarray(values["active_site_offset"], dtype=float),
            body_radius=float(values["body_radius"]),
            motif=values.get("motif", "") or "",
            flexible_tail=int(values.get("flexible_tail", 0) or 0),
            exit_offset=(
                np.asarray(values["exit_offset"], dtype=float)
                if values.get("exit_offset")
                else None
            ),
            exit_tail=int(values.get("exit_tail", 0) or 0),
            reference=values.get("reference", ""),
        )
        for name, values in document["effectors"].items()
    }
    motifs = {
        name: MotifPreference(
            name=name,
            base_weights={
                base.upper(): float(weight)
                for base, weight in values.get("base_weights", {}).items()
            },
            reference=values.get("reference", ""),
        )
        for name, values in document.get("motifs", {}).items()
    }
    return AssignedInputs(
        compositions=compositions,
        effectors=effectors,
        motifs=motifs,
        coarse_radii={
            key: float(value) for key, value in document.get("coarse_radii", {}).items()
        },
        structures=dict(document.get("structures", {})),
        scaffold_variants=dict(document.get("scaffold_variants", {})),
        structure_geometry=dict(document.get("structure_geometry", {})),
        substrate_rise=float(document.get("substrate_rise", 6.3)),
        observed_spread=float(document.get("observed_spread", 0.0)),
        surface_distance=float(document.get("surface_distance", 0.0)),
        occupancy_resolution=str(document.get("occupancy_resolution", "residue")),
        sampling={key: int(value) for key, value in document.get("sampling", {}).items()},
    )


def sha256_of(path: Path | str) -> str:
    """Digest of a configuration file, independent of line-ending convention.

    Carriage returns are normalised before hashing so that a checkout on
    Windows and one on Linux give the same digest for the same content.
    """
    data = Path(path).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def write_fit_record(paths: list[Path | str], output: Path | str) -> dict:
    """Record digests of the files the fit was run from.

    The fitted parameters and the assigned inputs are hashed together, since the
    assigned inputs carry most of the model's flexibility.
    """
    record = {
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "files": {str(path): sha256_of(path) for path in paths},
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return record


def entry_to_row(entry: CorpusEntry) -> dict[str, str]:
    row = {
        name.name: str(getattr(entry, name.name))
        for name in fields(entry)
        if name.name != "observed"
    }
    row["observed_profile"] = format_profile(entry.observed) if entry.observed else ""
    return row
