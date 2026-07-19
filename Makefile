.PHONY: setup setup-update predict play backtest backfill nightly test roi roi-settle

play:
	PYTHONPATH=. python scripts/ev_play.py

dashboard:
	PYTHONPATH=. python scripts/dashboard.py
	open outputs/dashboard.html 2>/dev/null || true

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

post-draw:
	bash scripts/monitoring/post_draw.sh

install-cron:
	mkdir -p ~/Library/LaunchAgents logs
	sed "s|__REPO__|$(CURDIR)|g" ops/com.lotto.postdraw.plist > ~/Library/LaunchAgents/com.lotto.postdraw.plist
	launchctl unload ~/Library/LaunchAgents/com.lotto.postdraw.plist 2>/dev/null || true
	launchctl load ~/Library/LaunchAgents/com.lotto.postdraw.plist
	@echo "Installed: post-draw routine runs Wed/Sat 22:30 (logs/post_draw.log)"


