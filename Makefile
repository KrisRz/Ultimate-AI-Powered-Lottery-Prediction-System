.PHONY: setup setup-update predict play backtest backfill nightly test roi roi-settle

play:
	PYTHONPATH=. python scripts/ev_play.py

setup:
	conda env create -f environment.yml || conda env update -f environment.yml
	@echo "Run: conda activate lotto-predict"

setup-update:
	conda env update -f environment.yml

predict:
	./predict_tonight.sh

backfill:
	PYTHONPATH=. python scripts/backfill_history.py

backtest:
	PYTHONPATH=. python scripts/validations/backtest.py --lookback 200 --step 5 --method frequency --compare random,probmap --seed 42

nightly:
	PYTHONPATH=. python scripts/monitoring/nightly_backtest.py

test:
	PYTHONPATH=. python -m pytest -q

roi-settle:
	PYTHONPATH=. python scripts/roi_ledger.py settle

roi: roi-settle
	PYTHONPATH=. python scripts/roi_ledger.py report


