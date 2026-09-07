IMAGE    ?= span:0.1.0
WORKBOOK ?= corpus/af4.xlsx
WORKDIR ?= $(CURDIR)
DOCKER  ?= docker run --rm -v $(WORKDIR):/work -w /work -u $(shell id -u):$(shell id -g) -e MPLCONFIGDIR=/tmp/matplotlib $(IMAGE)

.PHONY: image shell test count power calibration recovery structures substrate curate context fit link evaluate transfer invert residuals connector tale ortholog provenance pairing sensitivity bubble confinement leaveout stability margin substitution construct clean

image:
	docker build -t $(IMAGE) docker

shell:
	docker run --rm -it -v $(WORKDIR):/work -w /work $(IMAGE) bash

test:
	$(DOCKER) python -m pytest -q

count:
	$(DOCKER) python analysis/count_sources.py

power:
	$(DOCKER) python analysis/power_sim.py

calibration:
	$(DOCKER) python analysis/bootstrap_calibration.py

recovery:
	$(DOCKER) python analysis/synthetic_recovery.py

structures:
	$(DOCKER) python data/fetch_structures.py

substrate:
	$(DOCKER) python analysis/substrate_check.py

curate:
	$(DOCKER) python analysis/curate_kissling.py --workbook $(WORKBOOK)

context:
	$(DOCKER) python analysis/context_check.py

fit:
	$(DOCKER) python analysis/fit.py --jobs 8

link:
	$(DOCKER) python analysis/link_decision.py --jobs 8

evaluate:
	$(DOCKER) python analysis/evaluate.py

transfer:
	$(DOCKER) python analysis/anchor_transfer.py --max-chains 1000000

invert:
	$(DOCKER) python analysis/invert.py

residuals:
	$(DOCKER) python analysis/residuals.py

connector:
	$(DOCKER) python analysis/connector_spec.py

tale:
	$(DOCKER) python analysis/tale_check.py

ortholog:
	$(DOCKER) python analysis/ortholog_transfer.py

pairing:
	$(DOCKER) python analysis/strand_pairing.py

sensitivity:
	$(DOCKER) python analysis/fit_sensitivity.py --jobs 10

bubble:
	$(DOCKER) python analysis/bubble_sensitivity.py --jobs 10

confinement:
	$(DOCKER) python analysis/confinement_check.py

leaveout:
	$(DOCKER) python analysis/leave_one_out.py

stability:
	$(DOCKER) python analysis/mode_stability.py

margin:
	$(DOCKER) python analysis/architecture_margin.py

substitution:
	$(DOCKER) python analysis/ortholog_substitution.py

construct:
	$(DOCKER) python analysis/verify_construct.py --plasmid $(PLASMID)

map:
	$(DOCKER) python analysis/map_protospacer.py

effector:
	$(DOCKER) python analysis/measure_effector.py

series:
	$(DOCKER) python analysis/linker_series.py --effector tada8e --out results/linker_series_abe.json
	$(DOCKER) python analysis/linker_series.py --effector apobec1 --out results/linker_series_cbe.json

capture:
	$(DOCKER) python analysis/capture_sensitivity.py

biasseeds:
	$(DOCKER) python analysis/bias_seed_stability.py

seeds:
	$(DOCKER) python analysis/seed_stability.py

bias:
	$(DOCKER) python analysis/directional_bias.py

composition:
	$(DOCKER) python analysis/composition_sensitivity.py

clean:
	rm -rf cache results/*.json .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
