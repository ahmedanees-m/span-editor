# SPAN

[![tests](https://github.com/ahmedanees-m/span-editor/actions/workflows/tests.yml/badge.svg)](https://github.com/ahmedanees-m/span-editor/actions/workflows/tests.yml)
[![licence](https://img.shields.io/badge/licence-MIT-blue.svg)](LICENSE)
[![python](https://img.shields.io/badge/python-3.11-blue.svg)](pyproject.toml)

A geometric model of effector tethering in R-loop genome editors.

SPAN predicts the activity window of a tethered-effector editor from linker
statistics, steric exclusion and R-loop geometry. Three parameters are fitted to
two published editing profiles and held fixed; all other predictions are made
without adjustment. The package includes the model, a curated corpus of 50
published architectures with provenance for every entry, and the analysis
scripts that produce the reported results.

## Install

The model runs in the container described by `docker/Dockerfile`, which pins the
scientific stack. The repository is mounted rather than copied in, so editing
code does not rebuild the image.

```bash
make image
make test
```

To run outside the container, `environment.yml` describes the same dependencies
for conda, and the package installs with `pip install -e .` on Python 3.11.

Structure files are not redistributed. Fetch them from the PDB before running
anything that needs coordinates:

```bash
make structures
```

## Usage

Each analysis is a script under `analysis/` with a make target. Run the fit
first; the scripts that score predictions read `configs/fitted.yaml` and exit if
it is absent.

```bash
make fit          # fit three parameters on the training subset
make evaluate     # score the model and the baselines on the profile corpus
make leaveout     # score each position with its own coordinate withheld
make invert       # enumerate candidate tethers and their reachable positions
make connector    # connector geometry for the unreachable positions
```

`Makefile` lists every target, and each script accepts `--help` for its own
options. Results are written to `results/` as JSON.

Reproducing the published analysis in order:

```bash
make count calibration power        # corpus register and design sensitivity
make recovery                       # parameter recovery on generated profiles
make fit link sensitivity           # fitting and parameter identifiability
make evaluate leaveout transfer     # scoring
make ortholog substitution capture  # Cas12a analysis
make invert connector series        # reachable positions and connector geometry
make bias biasseeds seeds stability # sensitivity of the reported quantities
```

## Model

The editor is placed in the frame of the residue the effector is fused to. The
linker is sampled as a discrete worm-like chain of the persistence length
assigned to its composition class, with the effector placed at the far end in a
uniformly sampled orientation. The displaced non-target strand is sampled as a
second chain, pinned at nucleotides held by base pairing and sampled as bridges
between them. Conformations intersecting a coarse-grained map of the protein are
discarded during chain growth. Convolving the two distributions under a capture
radius gives a per-position capture probability, which is multiplied by the
effector's motif preference and passed through a saturating link.

Capture radius, displaced-strand persistence and link steepness are fitted.
Persistence lengths, coarse radii, active-site offsets, the structure chosen per
ortholog and the assignment of ordered strand segments are set from the
literature and recorded in `configs/assigned_inputs.yaml` with their sources.

Fitting uses per-position profile shape rather than window position, since window
position is invariant under any monotone link. Reporting scales differ between
publications, so profiles are compared as shapes with the multiplicative constant
solved by least squares.

Deposited strand coordinates are an input to the model rather than an output, so
predictions are scored with each position's own coordinate withheld
(`analysis/leave_one_out.py`).

## Layout

```
src/span_editor/   the model
  polymer.py       discrete worm-like chain samplers
  tether.py        tether sampling and effective-concentration fields
  geometry.py      anchor frames, substrate ensembles, excluded volume
  occupancy.py     capture probability, motif term, link, profiles
  model.py         assembly of a corpus entry into a prediction
  structures.py    anchor frames and strand coordinates from deposited structures
  fitting.py       grid search over the three fitted parameters
  baselines.py     marginal, substrate-only, distance and sterics, linear, modal linker
  metrics.py       window scoring, margins, cluster bootstrap
  invert.py        candidate tethers for a target window position
  synthetic.py     generated geometries for parameter recovery
  io.py            corpus, configuration and fit record
  power.py         detectable margin against the number of source publications
analysis/          one script per analysis, each with a make target
configs/           fitted parameters, search bounds, assigned inputs
data/              corpus, source register, structure list, data dictionary
results/           analysis output, as JSON
tests/             unit and integration tests
docker/            container definition
```

## Data

`data/architectures.tsv` holds 50 architectures from seven publications.
`data/sources.tsv` registers every publication examined, including those not
curated and the reason. `data/DATA_DICTIONARY.md` defines every column.

Linker length is recorded in residues read from construct maps rather than by
linker name, since XTEN denotes both a 16-residue and a 32-residue linker in
common use.

`data/architectures.example.tsv` holds generated rows used to exercise the
pipeline in tests and is not suitable for analysis.

## Baselines

Five, all implemented before the model was scored: a marginal baseline
predicting the corpus mean, a substrate-only baseline using R-loop exposure with
the tether ignored, distance plus sterics fitted with the same freedom as the
model, a linear-in-length baseline, and a modal-linker baseline for the inverse
problem.

Intervals resample source publications rather than entries, since entries cluster
by laboratory, publication, deaminase and scaffold. The construction is a
studentised wild cluster bootstrap. A percentile cluster bootstrap is available
and is not used for reporting: at these cluster counts it rejects a true null
roughly twice as often as its nominal rate.

## Citation

If you use this software or the corpus, please cite it using the metadata in
`CITATION.cff`.

## Licence

MIT. See `LICENSE`.
