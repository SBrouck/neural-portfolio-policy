PY ?= .venv/bin/python

.PHONY: env data train eval figures test all

env:
	uv venv -p 3.12 .venv && uv pip install -p $(PY) -r requirements.txt

data:      ## Yahoo Finance adjusted closes + FRED DTB3 -> data/
	$(PY) -m src.data_ingest

train:     ## 5 seeds, CPU, about 2 minutes -> results/runs/, results/training_log.csv
	$(PY) -m src.train --seeds 0 1 2 3 4

eval:      ## baselines + metrics tables -> results/*.csv
	$(PY) -m src.evaluate

figures:   ## -> figures/*.svg
	$(PY) -m src.figures

test:
	$(PY) -m pytest -q tests

all: data train eval figures
