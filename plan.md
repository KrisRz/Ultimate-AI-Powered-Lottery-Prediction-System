# PLAN — Analiza i plan ulepszenia „Ultimate AI-Powered Lottery Prediction System"

**Data analizy:** 2026-07-19
**Zakres:** pełna analiza lokalnego repo (stan lokalny > GitHub: `data/`, backtesty, Makefile, standalone skrypty są untracked)

---

## 0. Najważniejszy wniosek — przeczytaj to najpierw

**Aplikacja dziś NIE ma żadnej przewagi nad losowym typowaniem — i żaden model ML jej nie da.**

Twarde fakty z Twoich własnych wyników (`outputs/`):

| Dowód | Plik | Wynik |
|---|---|---|
| Jedyny backtest na >1 losowaniu (10 losowań) | `outputs/validation/validation_frequency_20250813_134920.json` | **random (0.7 avg trafień) POBIŁ model (0.5)** |
| „Zwycięski" backtest w `best_ensemble.json` | `outputs/results/best_ensemble.json` | oparty na **1 losowaniu** (`steps: 1`) — statystycznie bez wartości |
| Analiza 70 predykcji | `outputs/analysis/prediction_analysis_20250711_180309.json` | rozkład trafień = dokładnie rozkład hipergeometryczny losowego strzału (0 skilla) |
| Wszystkie backtesty łącznie | — | **ani jednego trafienia 3+** |
| Log produkcyjny | `logs/lottery.log` | modele się nie ładują → **system po cichu zwraca liczby losowe** („All model predictions failed, using random numbers") |

Losowania Lotto są niezależne i jednostajnie losowe — to matematyka, nie opinia. Sieci LSTM/CNN/XGBoost uczą się tu wyłącznie artefaktu sortowania kul (bo `y[:,i]` to posortowane kule) i szumu. **„Wygrywanie pieniędzy" przez przewidywanie liczb jest niemożliwe.**

**Co JEST możliwe (i na tym opiera się ten plan):**
1. **Maksymalizacja wartości oczekiwanej (EV) kuponu** — nie da się zwiększyć szansy trafienia, ale da się zwiększyć *wypłatę gdy trafisz*: granie liczb **niepopularnych** (ludzie masowo grają daty 1–31, „ładne" wzory, 7, 13…), więc trafiając niepopularną kombinacją dzielisz nagrodę z mniejszą liczbą osób. To jedyny udokumentowany naukowo „edge" w loterii (badania Cook & Clotfelter, Farrell i in.).
2. **Selekcja losowań po EV** — granie tylko przy rolloverach/Must-Be-Won, gdy EV na kupon jest najwyższe.
3. **Optymalizacja portfela kuponów** — pokrycie (coverage), brak duplikacji, dywersyfikacja — masz już zalążek w `new_predict.py` (`--optimize-coverage`).
4. **Uczciwy tracking ROI** — żeby wreszcie *wiedzieć*, ile system zarabia/traci.

Reszta planu: naprawa zepsutego kodu, radykalne odchudzenie, przebudowa celu z „przewidywania liczb" na „maksymalizację EV" + porządny dashboard.

---

## 1. Analiza stanu obecnego — część po części

### 1.1 Dane (backend danych)

**Co jest:**
- Gra: **UK National Lottery Lotto** (6 z 59 + bonus). Źródło: hardcoded CSV `https://www.national-lottery.co.uk/results/lotto/draw-history/csv` (`scripts/fetch_data.py:1147`).
- Pipeline: `download_fresh_data()` → `merge_data_files()` → `load_data()` z cache pickle. Solidnie zrobione: ETag/If-Modified-Since, retry z backoffem, walidacja schematu (6 unikalnych int 1–59), QA report do `outputs/results/data_quality.json`.

**Problemy (krytyczność malejąco):**
1. **Tylko ~60 losowań** (2025-01-15 → 2025-08-09), dane **~11 miesięcy nieświeże**. Oficjalny CSV daje tylko ~180 ostatnich losowań. Kod wielokrotnie odwołuje się do `data/lottery_data_1995_2025.csv`, **który nie istnieje** (`fetch_data.py:35`, `:1230`; `standalone_data_fetcher.py:21`) — `__main__` w `fetch_data.py` wywala się na starcie.
2. **`parse_balls()` potrafi sfabrykować losowe „wyniki"** z seeda zamiast odrzucić zły rekord (`fetch_data.py:63-72`) — cicha korupcja danych.
3. `scripts/standalone_data_fetcher.py` — martwy duplikat fetchera (0 importów).
4. Feature engineering istnieje **dwa razy**: aktywny `enhance_features()` w `fetch_data.py:124-300` (~40 cech) i martwy 682-liniowy `scripts/feature_engineering/feature_engineering.py` (tsfresh, importowany tylko przez własny `__init__.py`; tsfresh nawet nie jest w zależnościach).
5. `config/` — **11 plików, z czego 10 martwych** (0 referencji w kodzie). `config/data_sources.json` opisuje fikcyjną architekturę (API `lottery-data-provider.com`, które nie istnieje).

### 1.2 Modele ML (rdzeń „AI")

**Co jest:** ~5 470 linii w `models/` + ~4 000 w `models/utils/`: LSTM, CNN-LSTM, autoencoder, meta-model, XGBoost, LightGBM, CatBoost, GradientBoosting, KNN, Linear, ARIMA, Holt-Winters, ensemble.

**Problemy fundamentalne:**
1. **Zła definicja problemu:** wszystkie modele robią regresję 6 ciągłych wartości (posortowane kule) z lossem MSE. MSE nagradza „blisko liczbowo" (23 vs 24), co nie ma sensu w grze zbiorowej. Modele uczą się artefaktu sortowania, nie „wzorców".
2. **Niespójna denormalizacja:** LSTM trenuje `y/max(y)` per kolumna, a inferencja mnoży przez **59, 60 lub 49** zależnie od miejsca (`lstm_model.py:376,540`, `new_predict.py:317`, `compatibility.py:415`). Skale treningu i predykcji się nie zgadzają.
3. **Leakage:** wagi ensemble liczone inverse-MSE na kawałku **danych treningowych** (`train_models.py:1852-1866`); tuning używa `KFold(shuffle=True)` i `train_test_split` na szeregu czasowym (przyszłość przecieka do treningu).

**Problemy strukturalne:**
4. **12 klas wrapperów modeli NIGDY nie jest instancjonowanych** — nic w repo ich nie tworzy. Realna inferencja (`new_predict.py`) ładuje `.h5` bezpośrednio przez `keras.load_model` i **w ogóle nie importuje `models/`**.
5. **Podwójne trenery:** `scripts/improved_training.py` redefiniuje lokalnie `train_lstm_model`, `train_xgboost_model` itd., przykrywając wersje z `models/` — dwa rozjeżdżające się źródła prawdy.
6. **Kod, który nie może działać:** `LSTMModel.predict_next_draw` woła 3 nieistniejące metody (`lstm_model.py:330-351`); `LotteryEnsemble.predict` czyta `self.output_size`/`self.input_shape`, które nigdy nie są ustawiane (`ensemble.py:57,86,115`); `MetaModel` ma głowę `Dense(1)` fitowaną do 6-wymiarowego `y` (`meta_model.py:271,292`).
7. **4 niekompatybilne definicje `ensure_valid_prediction`** (dekorator vs 3 różne sanitizery) w 4 plikach; 3 różne drzewa utilsów (`models/utils`, `scripts/utils`, symlink między nimi).
8. **Zepsuty runtime:** zapisany LSTM oczekuje `(None,10,6)`, feature pipeline podaje `(None,30,15)` → `ValueError` → fallback na `random.sample` (log 2025-08-13). Do tego `keras.saving.pickle_utils` ImportError (niekompatybilność wersji Keras). **„Ensemble" w praktyce = 1 model LSTM, a często = czysty random.**
9. Martwy kod optuna (importowany, nigdy nie odpalany) w 4 modelach; deprecated API XGBoost ≥2.0 (`early_stopping_rounds` w `.fit()`).

### 1.3 Ewaluacja i backtesty

**Co jest dobre:** szkielet `scripts/validations/backtest.py` to poprawny walk-forward (okno przesuwne, out-of-sample, porównanie z randomem) — **to najlepszy kod w repo, budujemy na nim**.

**Problemy:**
1. Domyślny `--lookback 200` przy 60 rekordach → auto-shrink → **test na 1 losowaniu**. To wyprodukowało fałszywe „zwycięstwo" w `best_ensemble.json`.
2. Brak testu istotności statystycznej — nigdzie nie ma pytania „czy wynik odróżnia się od losowego z p<0.05".
3. Zepsute/mylące walidatory legacy: `prediction_validator.py:58-80` porównuje i-tą linię kuponu z i-tym historycznym losowaniem bez wyrównania czasowego (bez sensu); `performance_tracking.py:96-119` liczy RMSE/MAE na posortowanych kulach; `analyze_predictions.py:138-158` testuje predykcje na **losowo wygenerowanych** „wynikach".
4. Backtest woła `download_fresh_data()` na starcie → wyniki niereprodukowalne.

### 1.4 Testy

- Jedyny plik: `tests/benchmark_test.py` — benchmark wydajności, **nie przechodzi nawet importów** (4 ImportError: `train_lstm_model` to metoda klasy, `load_models` nie istnieje w `model_bridge`, złe nazwy funkcji) + 1 NameError.
- `pytest.ini` wymusza `--benchmark-compare`, który wywala się bez baseline'u.
- **Realnie: 0 działających testów.**

### 1.5 Entry pointy i architektura (backend aplikacji)

**~9 punktów wejścia, 1 realna ścieżka:**

| Entry point | Status |
|---|---|
| `predict_tonight.sh` → `scripts/new_predict.py` | ✅ **REALNA ŚCIEŻKA** (przez Makefile `make predict`) |
| `Makefile` (predict/backtest/nightly) | ✅ realny orkiestrator |
| `scripts/main.py` | cienki wrapper na `new_predict` — redundantny |
| `scripts/tensorflow_compatible_main.py` | duplikat standalone |
| `scripts/standalone_tensorflow_predictor.py` | duplikat standalone (używa nieistniejącego CSV) |
| `scripts/simple_runner.py` ⇄ `run_predictions.py` | wzajemne duplikaty (heurystyki, bez ML) |
| `simple_lottery.py` | zabawka: hardcoded hot/cold + random, 0 importów z projektu |
| `scripts/standalone_data_fetcher.py` | martwy |

**Frontend: nie istnieje.** Output to terminal + JSON/txt/png w `outputs/`. (Sekcja 2.6 — propozycja dashboardu.)

### 1.6 Środowisko i zależności

- **4 sprzeczne manifesty:** `requirements.txt` (kitchen-sink: sklearn przypięty 2×, TF+torch+prophet+mlflow+onnx…), `requirements_fixed.txt` (TF 2.15, bez metal), `environment.yml` (TF 2.13 + metal — konflikt), `installed.txt` (freeze).
- **`Miniconda3-latest-MacOSX-arm64.sh` (115 MB) + całe środowiska `conda-py311/` i `miniconda/` w repo.**
- `Makefile setup` tworzy env `lotto-predict` z `environment.yml`, ale `predict_tonight.sh` aktywuje **inny** env (`./conda-py311`) — setup i predict się nie zgadzają.
- `setup.py`: entry point `lottery-data=scripts.fetch_data:main` — funkcja `main` nie istnieje; `pyproject.toml` z placeholderem „Your Name".

---

## 2. Plan ulepszenia — fazy

### FAZA 1 — Wielkie sprzątanie (1–2 dni) 🧹
*Cel: z ~15 000 linii zostawić ~3 000, które faktycznie działają.*

- [x] **Usuń z repo:** `Miniconda3-*.sh` i `.venv/` skasowane; `conda-py311/` i `miniconda/` w `.gitignore` (zostają na dysku — to działający runtime). `__pycache__` i `outputs/` wyrzucone z indeksu gita.
- [x] **Usuń martwe entry pointy:** skasowane `scripts/main.py`, `tensorflow_compatible_main.py`, `standalone_tensorflow_predictor.py`, `standalone_data_fetcher.py`, `run_predictions.py`, `simple_lottery.py`, `_backup/` (został `scripts/simple_runner.py` jako zapasowy runner bez TF). Skasowany też klaster treningowy: `improved_training.py`, `train_models.py`, `model_bridge.py`, `analyze_data.py`, `performance_tracking.py`.
- [x] **Usuń martwy ML:** skasowane wszystkie pliki modeli poza `training_config.py` (używany przez `fetch_data`), `models/utils/`, `models/deployment/`; `models/__init__.py` wyczyszczony. Zostały artefakty w `models/checkpoints/`.
- [x] **Usuń martwe configi:** cały `config/` + `scripts/feature_engineering/` skasowane. Skasowany też zepsuty `tests/benchmark_test.py` i leaky `prediction_validator.py`; `pytest.ini` odchudzony.
- [ ] **Ujednolić utils:** przełożone na Fazę 4 (przy budowie pakietu `lottery/`); w `scripts/utils` zostały opcjonalne moduły z try/except fallbackami.
- [x] **Jeden manifest zależności:** został tylko `environment.yml`; `predict_tonight.sh` używa `./conda-py311`, z fallbackiem na env `lotto-predict` z `make setup`. Bonus: naprawiony bug połykania argumentów przez `source activate`.
- [x] Usunięte `setup.py` + `pyproject.toml` (broken entry point, placeholder autora) + `requirements.txt`, `requirements_fixed.txt`, `installed.txt`.
- [ ] Docelowa struktura (do zrobienia w Fazie 4):
  ```
  lottery/            # pakiet: data.py, features.py, ev.py, portfolio.py, backtest.py
  scripts/            # cienkie CLI: fetch, predict, backtest, nightly
  tests/
  data/  outputs/  Makefile  environment.yml  README.md  plan.md
  ```

### FAZA 2 — Napraw dane (1 dzień) 📊
- [x] **Backfill pełnej historii UK Lotto** — nowy `scripts/backfill_history.py` pobiera archiwum Merseyworld: `data/lotto_full_history.csv` = **3202 losowania 1994-11-19 → dziś** (z jackpotami i liczbą wygranych); `data/merged_lottery_data.csv` = **1125 losowań ery 59 kul** (od 2015-10-10) w formacie pipeline'u.
- [x] Odświeżone dane do najnowszego losowania (nr 3190, 2026-07-18). Z 60 → 1125 losowań (~19×).
- [x] Fabrykowanie liczb w `parse_balls()` usunięte — zły rekord = `ValueError`, nigdy generacja.
- [x] Widmowy `lottery_data_1995_2025.csv` usunięty z kodu; `__main__` w `fetch_data.py` używa merged CSV.
- [x] **Dane per tier:** oficjalny endpoint zwraca teraz **XML** (nie CSV) z pełnym rozbiciem nagród — `fetch_data.py` parsuje nowy format i akumuluje `data/prize_tiers.csv` (zwycięzcy + wypłaty per tier, rollover, szacowany następny jackpot, roll-down) przy każdym fetchu. Historyczne tiery: do douzupełnienia w Fazie 4 (beatlottery.co.uk ma per-draw breakdowny; wymaga scrapera z nagłówkami przeglądarki).
- [x] Pierwszy wiarygodny backtest: 185 losowań testowych, frequency 0.600 vs random 0.578 śr. trafień (oczekiwane losowe: 0.610) — **potwierdzony brak przewagi**, zgodnie z sekcją 0.

**⚠️ ZMIANA ZASAD GRY (odkryta przy backfillu):** od **7 czerwca 2026** każdy kupon Lotto gra w **DWÓCH rundach tej samej nocy** (dwie maszyny, dwa zestawy kul; wygrywasz w rundzie 1, 2 lub obu). Szansa na jakąkolwiek wygraną wzrosła z ~1:9,3 do ~1:4,9 przy tej samej cenie £2. Konsekwencje: (a) `lotto_full_history.csv` ma kolumnę `Round` (1/2), pipeline używa rundy 1, pełne dane są dostępne dla analiz; (b) silnik EV w Fazie 4 MUSI liczyć EV kuponu jako sumę po obu rundach; (c) stare tabele prawdopodobieństw tierów wymagają aktualizacji do nowej struktury nagród.

### FAZA 3 — Uczciwa ewaluacja (1–2 dni) ⚖️ ✅
*Zanim cokolwiek „ulepszysz", musisz umieć zmierzyć, czy to działa.*

- [x] Rozbudowany `backtest.py`:
  - [x] **twardy błąd** gdy kroków testowych < 30 (`--min-steps`) zamiast cichego shrinka do 1 losowania,
  - [x] test istotności: Monte Carlo vs dokładny rozkład hipergeometryczny (null = brak skilla), p-value dla avg trafień i 3+, bootstrap CI95, czytelny werdykt w konsoli („no edge over random…"),
  - [x] `--offline` + `--seed` → w pełni reprodukowalne runy; seed/offline zapisywane w JSON-ie wyników.
- [x] **Mylące walidatory usunięte:** `prediction_validator.py` i `performance_tracking.py` (Faza 1), `simulate_accuracy_test` wycięty z `analyze_predictions.py` (Faza 3).
- [x] **Tracking ROI:** nowy `scripts/roi_ledger.py` (`add` / `settle` / `report`) → `data/ledger.csv`; rozliczanie po **obu rundach** (format 2026), wypłaty per tier z realnych danych `prize_tiers.csv` (fallback: tabela szacunkowa); `make roi`. *Jedyna metryka, która mówi prawdę o „wygrywaniu pieniędzy".*
- [x] **31 testów pytest** (parser archiwum, `parse_balls`, ingestia XML, istotność backtestu, cykl ledgera) — zielone w ~2 s; workflow GitHub Actions (`.github/workflows/ci.yml`) gotowy na push. Bonus: naprawiony import-order bug (`setup_logging` funkcja vs submoduł — cykliczny import w `scripts/utils`).
- Wynik pierwszego pełnego pomiaru: frequency avg 0.692 (CI95 0.508–0.877) vs oczekiwane losowe 0.610, **p=0.195 → brak przewagi** (65 draws, seed 42, offline). System raportuje to wprost zamiast udawać skuteczność.

### FAZA 4 — Przebuduj „mózg": z przewidywania liczb na maksymalizację EV (3–5 dni) 💰
*Serce planu. Jedyna strategia z matematycznym uzasadnieniem.*

- [x] **4.1 Model popularności kombinacji** (`lottery/ev.py`): wagi liczb (daty 1–31 przegrywane częściej, szczyt 1–12; liczby >31 niedogrywane; boosty 7/3/11) + mnożniki wzorców (ciągi arytmetyczne ×8, potrójne sekwencje, kupon czysto „urodzinowy"). `popularity_ratio(line)` → `expected_cowinner_share` (model Poissona współzwycięzców jackpotu). **Do skalibrowania** na akumulujących się danych `prize_tiers.csv` (regresja liczby zwycięzców per tier).
- [x] **4.2 Kalkulator EV kuponu** (`lottery/ev.py::line_ev`): dokładne prawdopodobieństwa hipergeometryczne (zweryfikowane w testach z oficjalnymi kursami: jackpot 1:45 057 474, 5+bonus 1:7 509 579 itd.), stałe nagrody tierów z realnych danych 2026 (match5 £1000, match4 £50, match3 £24, match2 £5), jackpot dyskontowany współdzieleniem, **suma po OBU rundach**, minus £2. Uwaga: nagroda 5+bonus nieznana w danych (placeholder £250k — zaktualizować gdy się pojawi).
- [x] **4.3 Selektor losowań** (`should_play` + `scripts/ev_play.py`): warunki następnego losowania z `data/prize_tiers.csv` (szacowany jackpot, flaga roll-down z oficjalnego XML), próg EV (domyślnie 0 = graj tylko +EV), model roll-downu Must-Be-Won. Zweryfikowane: zwykłe £2M → SKIP (EV −£0.32), Must-Be-Won £12M → PLAY (EV +£1.27/kupon).
- [x] **4.4 Optymalizator portfela** (`lottery/portfolio.py`): sampling ważony ku niepopularnym liczbom, constrainty (overlap par ≤2, ≥2 liczby >31, pasmo sum 100–260, zakaz potrójnych sekwencji), greedy po EV. `make play`. OR-Tools CP-SAT: opcja na później, greedy wystarcza przy różnicach EV zdominowanych przez popularność.
- [x] **4.5 Degradacja ML:** wykonana de facto w Fazie 1 (wszystkie modele poza LSTM usunięte); stara ścieżka częstościowa (`predict_tonight.sh` → `new_predict.py`) zostaje jako legacy/sanity-check z uczciwym backtestem (p=0.195 → brak przewagi). Nowa główna ścieżka: **`make play`** (EV advisor). Ewentualna klasyfikacja BCE 59 prawdopodobieństw: świadomie odpuszczona — backtest potwierdził brak sygnału.

### FAZA 5 — Frontend / dashboard (2–3 dni) 🖥️ ✅
*Wybrany wariant: statyczny, samowystarczalny HTML (zero nowych zależności).*

- [x] `scripts/dashboard.py` → `outputs/dashboard.html` (`make dashboard`):
  - [x] **Dziś gramy?** — kafel werdyktu PLAY/SKIP + EV najlepszego kuponu (jackpot, roll-down z realnych danych),
  - [x] portfel z EV **przeliczanym na bieżące warunki** (nie zapisane z what-if) i score popularności,
  - [x] kafel ROI ledgera (wydane/wygrane/net/ROI%),
  - [x] backtest vs random: wykres skumulowanej średniej z hover-tooltipem, linia odniesienia no-skill 0.61, tabela z CI95 i p-value,
  - [x] świeżość danych (liczba losowań, ostrzeżenie gdy > 4 dni).
  - Zweryfikowany wizualnie w przeglądarce (light/dark, kolizje etykiet poprawione).

### FAZA 6 — Automatyzacja i higiena (1–2 dni) 🤖 ✅
- [x] Cron/launchd: `scripts/monitoring/post_draw.sh` (fetch → settle ledgera → dashboard → werdykt EV na następne losowanie) + szablon `ops/com.lotto.postdraw.plist`; instalacja jedną komendą `make install-cron` (śr./sob. 22:30, log w `logs/post_draw.log`). Alert mailowy przy +EV: do dodania w `post_draw.sh` (zalążek SMTP jest w `nightly_backtest.py`).
- [x] GitHub Actions: pytest na push (`.github/workflows/ci.yml`, Faza 3). Lint ruff: opcjonalny, odpuszczony świadomie.
- [x] Nowy README: uczciwy opis (EV-toolkit z wprost napisanym „nie przewiduje liczb"), quick start, struktura repo, oczekiwania finansowe.
- [x] `OPTIMIZATION_PLAN.md` zredukowany do notki „superseded by plan.md" (stare statusy „Completed" były sprzeczne z logami).
- [ ] Push na GitHub (gałąź `cleanup/phase-1` → PR/merge do `main`).

---

## 3. Kolejność wykonania i szacunek czasu

| Faza | Czas | Efekt |
|---|---|---|
| 1. Sprzątanie | 1–2 dni | −80% kodu, 1 env, 1 entry point, repo pushowalne |
| 2. Dane | 1 dzień | ~1100 losowań ery 59 kul + dane o zwycięzcach per tier |
| 3. Ewaluacja | 1–2 dni | wiarygodny backtest z p-value, ROI ledger, zielone testy |
| 4. EV-engine | 3–5 dni | popularność → EV → selekcja losowań → portfel kuponów |
| 5. Dashboard | 2–3 dni | panel decyzyjny zamiast printów w terminalu |
| 6. Automatyzacja | 1–2 dni | cron, CI, uczciwy README |

**Razem: ~2 tygodnie pracy wieczorami.**

## 4. Kryteria sukcesu

1. `make predict` działa bez fallbacku na random (0 błędów kształtu/importu w logu).
2. `pytest` zielony; CI na GitHubie.
3. Backtest na ≥200 losowaniach raportuje p-value vs random — i jest uczciwie pokazywany, nawet gdy mówi „brak przewagi".
4. Każdy wygenerowany portfel ma policzony EV i score niepopularności.
5. Ledger ROI pokazuje realny wynik finansowy w czasie.
6. Repo < 50 MB, jeden env, jeden entry point, README zgodny z rzeczywistością.

## 5. Uczciwe oczekiwania finansowe

UK Lotto ma zwrot ~45–50% stawek → **przeciętny gracz traci ~połowę każdej złotówki**. Strategia EV (niepopularne liczby + selekcja rolloverów/Must-Be-Won) podnosi *warunkową* wypłatę i w skrajnych losowaniach potrafi zbliżyć EV do dodatniego, ale wariancja jest ogromna — to optymalizacja „graj rzadko, graj mądrze, przegrywaj mniej", nie maszynka do zarabiania. Jedyny gwarantowany zysk z tego projektu to **umiejętności** (pipeline danych, backtesting, statystyka, EV-modeling) — które są wprost transferowalne np. do tradingu (patrz: TradePuls), gdzie realny edge jest przynajmniej teoretycznie możliwy.
