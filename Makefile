PY=python
PIP=pip

.PHONY: install bootstrap pilot lint

install:
	$(PIP) install -r requirements.txt

bootstrap:
	$(PY) scripts/bootstrap_db.py

pilot:
	$(PY) scripts/pilot_ingest.py --source cia-crest --limit 10

lint:
	@echo "No linter configured yet"
