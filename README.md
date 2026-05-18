# Bigspin: "Invisible Failures in Human–AI Interactions"

This is the code/data repository for the following report:

Potts, Christopher and Moritz Sudhof. 2026. Invisible failures in human--AI interactions. arXiv:2603.15423.

BiBTeX:

```
@unpublished{potts2026invisible,
	author = {Potts, Christopher and Sudhof, Moritz},
	note = {arXiv:2603.15423},
	title = {Invisible Failures in Human--{AI} Interactions},
	url = {https://arxiv.org/abs/2603.15423},
	year = {2026}}
```

* `paper_analyses.ipynb`: Produces all the tables, plots, and other analyses for the paper.
* `analysis_utils.py`: Supporting functions for `paper_analyses.ipynb`.
* `invisible_failures.mplstyle`: plot styling for the paper's figures.
* `persist_paper_stats.py`: Computes supporting stats for the the future-oriented experiments in the paper.
* `batch_quality_score.py`: Final annotation protocol.
* `taxonomy-tagging-code/`: The suit of tools needed to run a fully calibrated and validated annotation job.
* `data/wildchat_annotations_opus_v2.json.gz`: The main data file (100K annotated transcript).
* `data/interannotator/*`: Files for assessing interannotatora agreement.
* `data/persist/*`: Annotated files for the future-oriented experiments in the paper.
* `fig/`: Output directory for `paper_analyses.ipynb` containing all the figures in the paper.