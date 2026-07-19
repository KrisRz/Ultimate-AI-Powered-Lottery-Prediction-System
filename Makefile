.PHONY: setup setup-update predict backtest nightly

setup:
	conda env create -f environment.yml || conda env update -f environment.yml
	@echo "Run: conda activate lotto-predict"

setup-update:
	conda env update -f environment.yml

predict:
	./predict_tonight.sh

backtest:
	PYTHONPATH=. python scripts/validations/backtest.py --lookback 50 --method frequency --compare random,probmap

nightly:
	PYTHONPATH=. python scripts/monitoring/nightly_backtest.py


