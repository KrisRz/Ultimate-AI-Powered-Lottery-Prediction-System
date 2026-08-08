.PHONY: setup setup-update predict play dashboard backtest backfill sales nightly test roi roi-settle post-draw install-cron site-data site-data-check

# All python targets run inside the project runtime, so make works without an
# activated conda shell (launchd, cron, bare terminals). Override with e.g.
#   make test PY=python
PY ?= ./conda-py311/bin/python

play:
	PYTHONPATH=. $(PY) scripts/ev_play.py

dashboard:
	PYTHONPATH=. $(PY) scripts/dashboard.py
	open outputs/dashboard.html 2>/dev/null || true

# The public site's committed data snapshot. Run by a human, never by the
# collector bot: collect.yml pushes twice per draw, and wiring the exporter in
# there would publish unreviewed claims on a schedule.
site-data:
	PYTHONPATH=. $(PY) scripts/export_site_data.py

site-data-check:
	PYTHONPATH=. $(PY) scripts/export_site_data.py --check

setup:
	conda env create -f environment.yml || conda env update -f environment.yml
	@echo "Run: conda activate lotto-predict"

setup-update:
	conda env update -f environment.yml

predict:
	./predict_tonight.sh

backfill:
	PYTHONPATH=. $(PY) scripts/backfill_history.py

sales:
	PYTHONPATH=. $(PY) scripts/fetch_sales.py --validate

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
