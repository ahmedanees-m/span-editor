# Data dictionary

Two tables. `sources.tsv` registers the publications counted before curation
began. `architectures.tsv` is the corpus: one row per architecture per readout.

Every value in `architectures.tsv` is read from a primary figure or
supplementary table, and the panel it came from is recorded in the `panel`
column. Nothing is entered from a review or from a citing paper's description of
another paper's result.

## sources.tsv

| Column | Values | Meaning |
|---|---|---|
| `key` | short identifier | Used as `source_key` in the corpus and as the cluster label in bootstrap resampling |
| `citation` | free text | Author, journal, year, volume and pages where known |
| `doi` | DOI or blank | Present once the bibliographic record has been resolved against Crossref. A resolved record settles the author line and the venue; it says nothing about the values, which are read from the primary figure |
| `role` | `architecture`, `profile`, `external` | Architecture sources vary the tether and are the scarce resource. Profile sources supply per-position windows but vary deaminase or PAM variant. External sources are TALE-system work used as a check |
| `tether_variation` | free text, `none` | What the publication varies |
| `readout_class` | `amplicon_sequencing`, `functional_selection`, `plant_system`, `in_vitro_deamination`, `sanger_deconvolution` | Determines whether entries from this source may be fitted. Only amplicon sequencing may. A gel-based deamination assay on purified protein and a Sanger trace deconvolved by EditR are quantitative enough to give a window and not enough to fit against |
| `status` | `confirmed`, `verify_primary`, `to_retrieve`, `not_open_access`, `linker_lengths_unavailable`, `insertion_residues_unavailable`, `effector_geometry_missing`, `no_tether_variation` | Whether the entry has been checked against the primary, and where it has, what stopped it being curated. The three unavailable states all mean the same thing in practice: the geometry the model needs is in a figure panel rather than in the text or a table |
| `notes` | free text | Restrictions that apply to entries drawn from this source |

## architectures.tsv

| Column | Values | Meaning |
|---|---|---|
| `entry_id` | short identifier | Unique per row |
| `source_key` | key from `sources.tsv` | Bootstrap resampling is over this column, not over architectures |
| `editor` | free text | Construct name as published |
| `ortholog` | `SpCas9`, `SaCas9`, `SpG`, `SpRY`, `Cas12a`, `Nme2Cas9`, `IscB` | Cas protein family |
| `anchor_site` | key from `anchor_sites` in `configs/assigned_inputs.yaml` | Which residue of the Cas protein the linker leaves from. The names are per structure, so a site with no entry for the structure the row cites is an error rather than a fall back |
| `linker_residues` | integer | Contour length in residues. Zero means the source reports none. See below for where each value comes from |
| `linker_composition` | key from `configs/assigned_inputs.yaml` | Composition class, which sets the persistence length |
| `effector` | key from `configs/assigned_inputs.yaml` | Deaminase or other tethered domain |
| `label` | `tether_varied`, `effector_varied`, `scaffold_perturbed`, `both`, `none` | See below |
| `readout_class` | as in `sources.tsv` | Non-sequencing entries evaluate only. They never fit and never contribute to a width claim |
| `tier` | `A`, `B`, `C` | Evidence tier for the transfer hypothesis |
| `spacer_length` | integer | Protospacer length in nucleotides |
| `numbering` | `pam_distal`, `pam_proximal` | Which end position 1 sits at. Cas9 windows are quoted from the PAM-distal end and Cas12a windows from the PAM-proximal end |
| `structure_id` | PDB accession | Structure used for the anchor frame and excluded volume |
| `scaffold_variant` | key from `scaffold_variants`, or blank | Residue ranges the architecture removes from the Cas protein |
| `return_site` | anchor site, or blank | For a domain inserted into the protein rather than fused to a terminus, the residue the chain runs back to |
| `return_residues` | integer | Contour length of that return linker, in residues. Zero for a terminal fusion |
| `split` | `fit`, `held_out` | Terminal fusions in the SpCas9 family fit; everything else is held out. Set at curation and not revised |
| `panel` | free text | Figure or table the values were read from |
| `observed_profile` | `index:value;index:value;...` | Per-position editing, in protospacer index for the stated numbering |
| `observed_window` | `first-last`, or blank | The reported editing window as a span of protospacer positions, for sources that quote one instead of releasing a table |
| `observed_width` | integer, or blank | The reported width in nucleotides, where a source states a width without stating where the window sits |

### Linker length is recorded in residues, never by name

The name XTEN is used in the literature for a 16-residue linker in BE3 and for a
32-residue linker in the BE4max family. Curating by linker name would merge two
different tether lengths across a large part of the corpus, and tether length is
the primary input to the model.

Where each value comes from, since not every source states one:

| Source | Basis |
|---|---|
| Komor et al. 2016 | Stated in the text as GGS, (GGS)3, a 16-residue XTEN and (GGS)7 |
| Wang et al. 2019 | Stated as 16, 8 and 3 residues on one side and 32, 20 and 5 on the other |
| Tan et al. 2019 | Stated as the linker sequences themselves, PAP through PAPAPAP |
| Villiger et al. 2021 | The GGS and SGG linkers flanking the insertion are stated; the 32 residues on the terminal fusions are the ABEmax architecture |
| Kissling et al. 2025, Huang et al. 2019 | Not stated. The 32 residues come from the ABEmax and ABE8e architecture, which those papers name rather than describe |
| Chen et al. 2022 | Not stated in the paper, read from the deposited sequence of the authors' own plasmid, Addgene 193640: the deaminase, then SGSETPGTSESATPES, then LbCas12a |

The distinction matters for width and not for position. The reachable-position
map shows the predicted window mode is flat in contour length from 4 to 48
residues at a terminus, so no positional result rests on an assumed length. The
width results do rest on it, and they come from the one source that states its
lengths.

### The three labels

`tether_varied` covers a change of linker length, composition or attachment
terminus with the Cas protein and effector unchanged.

`effector_varied` covers a change of deaminase active site or truncation with
the tether unchanged. The model has no deaminase-identity input, so predicting
no window change across these entries is close to automatic. The label exists to
catch one failure mode: deaminase identity leaking in through structure
selection, active-site offsets or the motif term.

`scaffold_perturbed` covers a change to the Cas protein itself. Inlays at 1054
and 1246 sit in RuvC and the PAM-interacting domain, replacement of HNH removes
part of the conformational checkpoint, and circular permutants reorganise the
fold. All of these alter DNA binding, R-loop kinetics and residence time, which
enter the model through the global activity scale.

Every architecture in this corpus that moves the effector off a terminus carries
this label, and that is not an accident of curation. A tether cannot leave the
protein anywhere except a terminus without the backbone being cut, so anchor
transfer and scaffold perturbation are confounded by construction in the
published literature. The subset of anchor transfers with the Cas protein
untouched is empty, and anchor-transfer results are reported as what they are
rather than split into a clean half that does not exist.

`both` covers entries where tether and effector change together, such as a
linker removal that accompanies a new deaminase.

### Windows reported as spans

Most architecture work does not release a per-position table. It says where the
window sits, or how wide it is, and compares that with the ordinary terminal
fusion measured in the same experiment. `observed_window` and `observed_width`
carry those, and an entry may have either, both or neither.

An entry with neither is still curated when it is the reference its source
compares everything else against, or when the reported result is that the
construct does not edit at all.

### Domains inserted rather than fused

A deaminase put in place of a domain the editor already carried, or inlaid into
a surface loop, is held at both ends. `anchor_site` names the residue the chain
leaves from, `return_site` the residue it runs back to, `linker_residues` and
`return_residues` the two contour lengths. The effector entry for such a row is
measured from its other junction, since the incoming chain arrives at the end
the terminal-fusion geometry departs from.

### Profiles

Values are per-position editing as published, on whatever scale the source
reports. Comparisons normalise within a profile, so the absolute scale does not
need to be uniform across sources. Positions absent from a published panel are
left out of the string rather than entered as zero.

## Configuration

`configs/params.yaml` holds the fitted parameters and two conventions that would
otherwise be implicit.

`link.reference_radius` fixes what the saturation index is measured against. The
index is the link steepness multiplied by the largest capture probability in the
training subset, and it exists because the steepness on its own is not
comparable between runs: capture probability is a small number whose size
depends on how densely the tether and the substrate were sampled. The index is
the argument of the exponential at the peak, so it says directly whether the
link is doing anything. Well below one the link is close to linear.

The peak is always taken at the reference radius and never at whichever radius
the search happens to be visiting. An adaptive reference would make the index
depend on the grid rather than on the model, and the same fit would then report
different indices under different searches.

`link.profile_scale` fixes how the reporting scale of a source is removed before
profiles are compared. Dividing by the peak is the obvious choice and is what
reading a window off a published figure amounts to, but the peak is one order
statistic and noise inflates it, which flattens everything else. On generated
profiles that biases the recovered saturation index down by about a third and
puts the generating value outside the interval. Solving for the multiplicative
constant that minimises the squared error uses every position instead. The
constant is obtained in closed form per entry and never carried between entries,
so it adds nothing to the fitted parameter count.

`configs/assigned_inputs.yaml` holds the inputs that are assigned rather than
fitted: persistence lengths, coarse radii, effector geometry, per-structure chain
assignments and the choice of structure per ortholog. Every entry carries a
status. `measured` means it was taken from the cited structure, `to_verify` that
the primary citation is still to be attached, and `provisional` that the value is
dimensionally reasonable but has been neither read nor measured.

## Example rows

`architectures.example.tsv` holds generated rows used to exercise the pipeline
and the tests. The profiles in it were not measured and must not appear in any
analysis that carries a claim.
