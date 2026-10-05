# Segment support for rhizotron root masks (Munoz et al. 2026)
#
#   make venv            virtual environment + dependencies
#   make example         Figure S3 case (ArUco 109, 24 DAT)
#   make example-38dat   same rhizotron at 38 DAT
#   make test            recompute the example tables, compare with the paper's fingerprints
#   make notebook        run the notebook headlessly and keep its outputs
#   make lab             open the notebook in JupyterLab
#   make clean           remove outputs and caches, keeps the venv
#
# Another interpreter:  make venv PYTHON=python3.12

PYTHON ?= python3
VENV   ?= venv
BIN    := $(VENV)/bin
PY     := $(BIN)/python

.PHONY: help venv example example-38dat test notebook lab clean distclean

help:
	@sed -n '2,11p' Makefile | sed 's/^# \{0,1\}//'

$(BIN)/activate:
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install --upgrade pip

venv: $(BIN)/activate
	$(PY) -m pip install -r requirements.txt
	@echo "activate with:  source $(BIN)/activate"

example: venv
	$(PY) examples/run_example.py

example-38dat: venv
	$(PY) examples/run_example.py --date 2025-04-21

test: venv
	$(PY) -m pytest tests/ -q

notebook: venv
	$(PY) -m pip install -q nbclient
	cd examples && ../$(PY) -c "import nbformat; from nbclient import NotebookClient; \
	  p='segment_support_example.ipynb'; nb=nbformat.read(p, as_version=4); \
	  NotebookClient(nb, timeout=1800, kernel_name='python3').execute(); nbformat.write(nb, p); print('notebook executed')"

lab: venv
	$(PY) -m pip install -q jupyterlab
	$(BIN)/jupyter lab examples/segment_support_example.ipynb

clean:
	rm -rf examples/output .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	find . -name .ipynb_checkpoints -type d -prune -exec rm -rf {} +

distclean: clean
	rm -rf $(VENV)
