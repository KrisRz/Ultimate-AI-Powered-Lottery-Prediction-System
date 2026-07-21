.PHONY: setup setup-update predict play dashboard backtest backfill nightly test roi roi-settle post-draw install-cron

# All python targets run inside the project runtime, so make works without an
# activated conda shell (launchd, cron, bare terminals). Override with e.g.
#   make test PY=python
PY ?= ./conda-py311/bin/python

play:
	PYTHONPATH=. $(PY) scripts/ev_play.py

dashboard:
	PYTHONPATH=. $(PY) scripts/dashboard.py
	open outputs/dashboard.html 2>/dev/null || true

setup:
	conda env create -f environment.yml || conda env update -f environment.yml
	@echo "Run: conda activate lotto-predict"

setup-update:
	conda env update -f environment.yml

predict:
	./predict_tonight.sh

backfill:
	PYTHONPATH=. $(PY) scripts/backfill_history.py

backtest:
	PYTHONPATH=. $(PY) scripts/validations/backtest.py --lookback 200 --step 5 --method frequency --compare random,probmap --seed 42 --offline

nightly:
	PYTHONPATH=. $(PY) scripts/monitoring/nightly_backtest.py

test:
	PYTHONPATH=. $(PY) -m pytest -q

roi-settle:
	PYTHONPATH=. $(PY) scripts/roi_ledger.py settle

roi: roi-settle
	PYTHONPATH=. $(PY) scripts/roi_ledger.py report

post-draw:
	bash scripts/monitoring/post_draw.sh

install-cron:
	mkdir -p ~/Library/LaunchAgents logs
	sed "s|__REPO__|$(CURDIR)|g" ops/com.lotto.postdraw.plist > ~/Library/LaunchAgents/com.lotto.postdraw.plist
	launchctl unload ~/Library/LaunchAgents/com.lotto.postdraw.plist 2>/dev/null || true
	launchctl load ~/Library/LaunchAgents/com.lotto.postdraw.plist
	@echo "Installed: post-draw routine runs Wed/Sat 22:30 (logs/post_draw.log)"
