# PLAN — Analiza i plan ulepszenia „Ultimate AI-Powered Lottery Prediction System"

**Data analizy:** 2026-07-19
**Zakres:** pełna analiza lokalnego repo (stan lokalny > GitHub: `data/`, backtesty, Makefile, standalone skrypty są untracked)

---

## STATUS: PRODUKCJA — działa samodzielnie (aktualizacja 2026-07-19)

Fazy 1–6 wdrożone i zmergowane do `main` (PR #1–#5). Faza 7 (pętla danych) **uruchomiona i pracuje**:
- 🟢 Kolektor chmurowy (GitHub Actions) zbiera dane po każdym losowaniu (śr./sob. 21:45 UTC) + retry (czw./niedz. 6:00) + commit danych do repo.
- 🟢 Watchdog (czw./niedz. 12:00) alarmuje mailem, gdy losowanie nie zostało zebrane (dane z oficjalnego XML są nieodwracalne po ~72 h).
- 🟢 Maile przez SMTP potwierdzone na **obu** torach (chmura + lokalny Mac); alert tylko przy werdykcie PLAY lub awarii.
- 🟢 Tor lokalny launchd (`post_draw.sh`): pull danych z chmury → settle ledgera (prywatny, poza repo) → dashboard.
- 🟢 58 testów. Koszt: £0/mies. Pierwsza automatyczna zbiórka: środa 2026-07-22 22:45 czasu PL.

**➡️ STAN 2026-07-25:** kalibracja popularności ZROBIONA przedterminowo (backfill 1126 historycznych rozbić nagród z lottery.co.uk zdjął blokadę „czekaj 25+ losowań") — wagi `number_weight` skalibrowane na danych, nagroda 5+Bonus poprawiona na £1M. Szczegóły w Fazie 7 niżej. Otwarte: zmienne Match 3/Match 2 (do weryfikacji przy większej liczbie oficjalnych losowań 2-rundowych) + roll-down watch. Appka pracuje sama — użytkownik czeka na mail „PLAY".

**✅ CRON POTWIERDZONY (2026-07-25):** scheduler GitHub Actions odpalił **sam** po raz pierwszy — `Collect draw data` (event=`schedule`) śr. 2026-07-22 22:42 UTC + retry czw. 08:41 UTC, `Collection watchdog` czw. 13:41 UTC, wszystkie sukcesem. Commit `1262502 data: collect draw 2026-07-22` na main (pierwszy realny commit z workflow). Losowanie 3191: 12 wierszy tierów, jackpot £2M niezerowy w historii, JackpotWins=0 (rollover). Tor lokalny launchd też zebrał to samo (`logs/post_draw.log` śr. 22:52). Pętla danych działa w pełni bez interwencji.

---

## AUDYT 2026-07-21 — głęboka analiza i naprawy

Trzyagentowy audyt (silnik EV, pipeline danych, ewaluacja). Rdzeń matematyczny zweryfikowany numerycznie jako poprawny (hipergeometryka = oficjalne kursy, Poisson współzwycięzców, estymator sprzedaży 8,38 mln spójny między tierami, mapowanie tierów→rund spójne wszędzie — kamień „zweryfikować mapowanie" z Fazy 7 zaliczony przedterminowo).

**Rozstrzygnięcie kluczowej niewiadomej — semantyka jackpota:** oficjalny komunikat Allwyn: *„The jackpot will be shared across both rounds, while all other prize tiers will continue to offer fixed cash prizes, paid per round."* Jackpot = **jedna pula na całe losowanie** (nie per runda). Stary kod liczył roll-down 2× za wysoko i lekko zawyżał jackpot EV. Naprawione w `lottery/ev.py` (λ współzwycięzców × rundy; roll-down: rundy się skracają, per-kupon = e^(−2N·P_J)·J/N) + test regresyjny.

**Naprawione tego dnia:**
1. `roi_ledger.py`: kolumny bool/str po round-tripie CSV (pandas ≥2.2 rzucał TypeError w `settle` — to było źródło czerwonego CI na main) + zakaz rozliczania losowania 2026+ przy tylko 1 rundzie w danych.
2. `lottery/ev.py`: linia referencyjna była ciągiem arytmetycznym [32,34,36,…] (kara ×8 niwelowała niepopularność, ratio 1.01 zamiast 0.13) → teraz [32,34,37,39,41,43], ratio ≈0.13; klucz w werdykcie `jackpot_per_round`→`jackpot_event_pool`.
3. `dashboard.py`: używał własnej kopii warunków (15 mln kuponów zamiast estymowanych 8,4 mln → możliwy rozjazd mail PLAY vs dashboard SKIP) → teraz importuje `next_draw_conditions()` z `ev_play` (jedno źródło prawdy, to samo co alert mailowy).
4. `ev_play.py`: flaga Must-Be-Won odporna na NaN (`bool(NaN)` dawał fałszywy PLAY), guard na pusty `prize_tiers.csv`, `latest.json` zapisywany też przy SKIP (koniec pętli „run make play" na dashboardzie).
5. `fetch_data.py`: dopisywane do historii losowania miały `Jackpot: 0` (obliczony `jackpot_raw` nieużyty!) → teraz jackpot z `next_jackpot_estimate` poprzedniego losowania + realne `JackpotWins` z tierów. Ważne dla kalibracji Fazy 7 — dane są nieodwracalne.
6. `collect.yml`: krok walidacji świeżości (fetch „udaje się" nawet przy braku sieci — teraz brak oczekiwanego losowania = czerwony run).
7. `collection_watchdog.py`: uszkodzony/pusty CSV alarmuje mailem zamiast crashować przed wysyłką.
8. `post_draw.sh`: nieudany rebase robi `--abort` (wcześniej zostawiał repo mid-rebase i zabijał wszystkie kolejne runy).
9. `Makefile`: `PY ?= ./conda-py311/bin/python` — `make test/backtest/nightly` działają bez aktywowanej condy; `make backtest` z `--offline` (reprodukowalność).
10. Sprzątanie: usunięte `predictions.py`, `simple_runner.py`, `compare_runs.py`, `analyze_predictions.py` (~1800 linii martwego/pseudonaukowego kodu) + `trained_models.pkl.backup` (4,2 MB); untracked `logs/lottery.log`, `data/.download_state.json`, `data/lotto-latest.xml` (wiecznie brudne drzewo).

**Otwarte po audycie (do Fazy 7):**
- Model roll-downu (uniform split) niezweryfikowany do pierwszego realnego Must-Be-Won; w danych 3190 jackpot zresetował się do £2M przy 0 zwycięzcach tier 1 i rollover=False — obserwować.
- `popularity_ratio` nieznormalizowane po mnożnikach wzorców (globalna λ zawyżona) — poprawić przy kalibracji wag.
- `SUM_BAND` górna granica 260 w portfelu ucina najlepsze niepopularne linie (sumy ~300) — rozluźnić przy kalibracji.
- Werdykt PLAY liczony na linii referencyjnej; po zbudowaniu portfela warto walidować `min(EV) ≥ próg`.

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
- [x] Push na GitHub: gałąź `cleanup/phase-1` wypchnięta (CI uruchomi się automatycznie). Merge do `main`: decyzja użytkownika (PR: https://github.com/KrisRz/Ultimate-AI-Powered-Lottery-Prediction-System/pull/new/cleanup/phase-1).

---

### FAZA 7 — Pętla danych i tuning (następne 2–3 miesiące) 🔄
*Uruchomiona 2026-07-19. Cel: za 2–3 miesiące skalibrowany model EV i dyscyplina grania wyłącznie przy PLAY.*

**Architektura zbierania (£0/mies., dwa niezależne tory dla niezawodności):**
- **Chmura — główny** (`.github/workflows/collect.yml`): fetch po każdym losowaniu + retry następnego ranka, idempotentny zapis (dedupe `draw_number+round+tier`), **commit danych do repo** (dane publiczne = wersjonowanie + backup w gicie), werdykt EV, mail przy PLAY. Watchdog (`watchdog.yml` + `collection_watchdog.py`): poranek po losowaniu sprawdza, czy dane weszły — brak = czerwony run + mail alarmowy, póki dane z XML jeszcze do uratowania.
- **Lokalny — zapasowy** (launchd `com.lotto.postdraw`, śr./sob. 22:30): `git pull` danych z chmury → settle ledgera (prywatny, poza repo) → dashboard. Mac musi być włączony/uśpiony; jeśli leżał offline, po `git pull` i tak ma komplet z chmury.
- Świadomie **bez bazy danych**: ~110 wierszy/mies., cała historia 170 KB — CSV w gicie jest dla tej skali niezawodniejszy niż hostowany Postgres (analiza w rozmowie 2026-07-19). Ewentualny SQLite dopiero gdy zapytania kalibracyjne urosną.
- Każde losowanie dodaje 12 wierszy do `data/prize_tiers.csv` (zwycięzcy + wypłaty per tier × 2 rundy, rollover, następny jackpot).

**Co już skalibrowane z danych (2026-07-19):**
- [x] Liczba sprzedanych kuponów: **~8,4 mln/losowanie** estymowane z liczby zwycięzców tierów match-2/3/4 (`estimate_tickets_sold`) zamiast założonych 15 mln — EV advisor używa tego automatycznie.

**Co skalibrowane 2026-07-25 (przedterminowo — mieliśmy dane):**
- [x] **Backfill historycznych rozbić nagród** — `scripts/backfill_prize_tiers.py` ściąga z `lottery.co.uk/lotto/results-DD-MM-YYYY` (liczby zwycięzców per tier per runda, zgodne CO DO SZTUKI z oficjalnym feedem). Cała era 59-kulowa: **1126 losowań** w `data/prize_tiers_history.csv`. To zdjęło blokadę „czekaj 25+ losowań".
- [x] **Kalibracja wag popularności** — `scripts/calibrate_popularity.py` na 1139 rundach: losowania „urodzinowe" (dużo liczb ≤31) mają **+78% więcej zwycięzców/kupon** (monotonicznie 0,75→1,84; korelacja z modelem +0,51). Wagi odzyskane fitem forward-modelu (od-tłumienie ×2) → wpisane do `number_weight`: **≤12: 1,23 / 13–31: 1,10 / >31: 0,83** (było 1,35/1,20/0,72 — model był ~2× za agresywny). Słownik „lucky" usunięty (boosty w granicach szumu).
- [x] **Match 5+Bonus = £1 000 000** (potwierdzone w oficjalnych i historycznych danych) — zastąpiło placeholder £250k. Efekt: próg opłacalności non-MBW spadł £9,2M→£4,76M.

**Co skalibrowane 2026-07-28 — KRYTYCZNY FIX:**
- [x] **Match 3 / Match 2 = £10 / £1** (nie £24 / £5). Zagadka „zmiennych" nagród rozwiązana: losowanie **3190 (2026-07-18) było ROLL-DOWNEM** — strona źródłowa podaje wprost `"£10 Rolldown Prize: £24"` i `"£1 Rolldown Prize: £5"` (stąd sklejki `1024`/`15` w scrapie). Kod czytał tę jedną, nietypową obserwację jako normę. Baza £10/£1 występuje w **12 z 13** losowań ery dwurundowej. Test zdrowego rozsądku: przy £24/£5 same stałe tiery zwracałyby 90% wpływów; przy £10/£1 → 36% + jackpot ≈ realne ~50%.
  - **Skutek błędu:** EV zawyżone o **£1,07/kupon**; próg opłacalności £4,76M zamiast £30,2M. Przy jackpocie £4,44M i rosnącym rolloverze advisor był o jedno losowanie od fałszywego maila **PLAY** przy realnym EV ≈ −£1,03.
  - **Fix:** stałe bazowe w `lottery/ev.py` + nowe `calibrate_fixed_prizes()` — nagrody liczone z `data/prize_tiers.csv` **medianą** przy każdym uruchomieniu (roll-downy są mniejszością, więc nie ruszą mediany; średnia albo pojedyncze losowanie wpuściłyby boost z powrotem). `FixedPrizes` wpięte w `DrawConditions`, advisor i dashboard drukują użyte nagrody + próg opłacalności. Tabela fallback w `roi_ledger.py` też poprawiona (przy okazji 5+bonus 250k→£1M). 81 testów zielonych.
- [x] **Roll-down zwalidowany** (na 3190): boost poszedł WYŁĄCZNIE na Match 3 (+£14 × 169 438) i Match 2 (+£4 × 1 756 390) = **£9,40M** rozdane przy puli must-be-won **£9,56M** (98%); Match 4/5 bez zmian. Nasz człon `J/N` daje £1,28/kupon vs £1,26 realnie wypłacone → model dobry, zostaje. Uwaga: roll-down jest ślepy na popularność (stałe boosty), więc niepopularna linia nic tu nie zyskuje.
- [x] **Nowe `break_even_jackpot()`** — advisor/dashboard mówią wprost, ile musiałby wynieść jackpot: **£30,2M zwykłe / £9,19M przy Must-Be-Won**. Czyli realny +EV istnieje tylko przy MBW.

**Deep audit 2026-07-28 — co mówią dane historyczne (1126 losowań):**
- **Ile realnie jest okazji +EV: ~9 rocznie z 104 losowań.** Limit przetoczeń wynosi **5** — odczytany z danych, nie z regulaminu: od 11.2018 żadna seria nie przekroczyła 5, a wszystkie **70** losowań, które doszły do 5, rozstrzygnęły się na miejscu (53 roll-down, 17 trafiony jackpot). To daje **9,1 losowań Must-Be-Won rocznie, 1 na 12**, a **67 z 70 (96%)** miało jackpot ≥ progu £9,19M → **8,7 okazji +EV/rok**, plus ~0,9/rok zwykłych losowań powyżej £30,2M (rekord ery £52,9M, p99 £28,7M). Wniosek operacyjny: **czekamy na Must-Be-Won, resztę pomijamy** — advisor dostaje flagę z wyprzedzeniem z oficjalnego XML (`next_jackpot_roll_down`), a od 2026-07-28 sam liczy też, ile losowań zostało do limitu. Zweryfikowane end-to-end: przy £12,8M + roll-down (mediana MBW = £12,0M) advisor daje **PLAY, EV +£0,50/linia**.
  - *Uwaga metodologiczna:* wykrywacz „podbite niskie tiery" znajduje 93 losowania, ale tylko 70 to Must-Be-Won z limitu; pozostałe **36 (~4,7/rok) to promocyjne podbicia** (30 z 36 w soboty), które podnoszą EV o ~£0,37/linię — za mało, by odwrócić −£1,08. Model wycenia tylko te pierwsze.
- **Tabela nagród ery 1-rundowej potwierdzona** na 1113 obserwacjach: Match5 £1750 ✓, Match4 £140 ✓, Match3 £30 (IQR 25–30) ✓, Match2 £0 (darmowy los) ✓, 5+bonus £1M ✓ — `PRIZES_OLD_RULES` w ledgerze zgadza się z danymi.
- **Estymator N stabilny**: zgodność między tierami M4/M3/M2 w granicach 1,14–1,30×; mediana ery 8,5–9,1M linii, era 2-rundowa 6,5–7,5M. `DEFAULT_TICKETS_SOLD` poprawione **15M → 7,5M** (15M było zgadywane, 1,8× za dużo).
- **Kalibracja popularności trzyma się out-of-sample**: efekt „urodzinowy" widoczny w obu połowach historii. ✅ **Refit na nowszych danych (2026-07-28) potwierdził stabilność wag** — `--last-draws 500`: 1,222 / 1,123 / 0,822, `--since 2023`: 1,212 / 1,132 / 0,820, `--last-draws 300`: 1,218 / 1,133 / 0,816, wobec 1,23 / 1,10 / 0,83 w kodzie. Różnice w granicach szumu fitu (<3%), wagi zostają bez zmian. Wcześniejsza obawa „efekt silniejszy w nowszej połowie (+67% vs +143%)" była artefaktem zgrubnego estymatora (mediana znormalizowanych zwycięzców w kubełkach), nie regresji — sama regresja nie widzi dryfu. `calibrate_popularity.py` ma teraz `--since` / `--last-draws`, żeby ten test dało się powtórzyć jedną komendą.
- ✅ **Model popularności znormalizowany** (2026-07-28): `E[popularity_ratio]` po losowych liniach wynosił **1,046** zamiast 1,0 — mnożniki wzorców dokładały masę, której nic nie odbierało, więc `tickets_sold × ratio / TOTAL_COMBOS` opisywał 4,6% więcej kuponów, niż sprzedano. `POPULARITY_NORMALIZATION` liczone **dokładnie** (nie próbkowaniem) i przy każdym imporcie, więc rekalibracja wag nie zostawi nieaktualnej stałej: DP po 59 kulach ze stanem (wybrane, długość serii, czy była trójka) zamiast enumeracji 45 mln linii, plus 319 ciągów arytmetycznych wyliczonych wprost. DP zweryfikowane brute-force'em na loterii 4 z 12 (zgodność do 1e-12), a Monte Carlo po normalizacji daje 1,0001 ± 0,0010.
- **SUM_BAND (100–260) jest nieszkodliwy** — obcina 75% linii w całości >31, ale najlepsza dopuszczalna linia ma popularity 0,330, dokładnie tyle co najlepsza bez constraintu (po kalibracji wszystkie liczby >31 mają tę samą wagę, więc optimum jest zdegenerowane). Stary punkt „SUM_BAND obcina najlepsze linie" — nieaktualny, zamknięty.
- **Dane czyste**: 3206 wierszy / 3192 losowania, 0 duplikatów, 0 luk w numeracji, 0 błędnych kul, komplet 2 rund w każdym losowaniu ery dwurundowej; χ² równomierności 59 kul p=0,98 (brak sygnału, zgodnie z oczekiwaniem).
- **Naprawione przy okazji:** (a) przebiegi what-if (`--jackpot/--roll-down/--tickets`) **nadpisywały `latest.json`** — plik, z którego ledger bierze linie „naprawdę zagrane"; teraz what-if nic nie zapisuje (+6 testów); (b) 4 fałszywe WARNINGi w każdym produkcyjnym logu (`scripts/utils` ładował martwe moduły ery TF — `cross_validation` sięga po skasowany `scripts.train_models`, reszta po brakujący `psutil`); (c) watchdog podpowiada teraz scraper archiwum jako drugą szansę na odzyskanie przegapionego losowania.

**Próba generalna pełnej pętli 2026-07-28** (ledger ma 0 kuponów, więc ścieżka PLAY nigdy nie przeszła end-to-end; przećwiczona w piaskownicy na prawdziwych danych, warunki ustawione jak przed Must-Be-Won):
- Przeszło: advisor → PLAY z portfelem → `roi_ledger add --from-latest` → `settle` na realnych wynikach 3190 (jedna linia trafiła 2 → £5 z **realnej, podbitej** nagrody roll-downu, nie z tabeli bazowej) → raport ROI −50% → **dashboard z niepustym kaflem ledgera** (ta ścieżka renderowała się pierwszy raz w historii projektu).
- ❌ **Znaleziona dziura:** chmura (`collect.yml`) uruchamia **wyłącznie** `ev_alert.py`, nigdy `ev_play.py`, a `outputs/` jest w `.gitignore` — w chmurze nie istnieje żaden `latest.json`. Mail PLAY mówił „uruchom `make play`" i „`add --from-latest`", czyli odsyłał do maszyny, przy której akurat Cię nie ma. Ścieżka odpalająca się ~9 razy w roku była bezużyteczna dokładnie w sytuacji, dla której powstała.
- ✅ **Naprawione:** `ev_alert.build_alert()` sam buduje portfel i wkłada do maila linie, datę losowania, koszt, próg opłacalności i gotową do wklejenia komendę `roi_ledger add --draw-date ... --lines "..."`. Ziarno z daty losowania, więc wieczorny przebieg i poranny retry proponują **te same** linie zamiast dwóch różnych. Błąd budowy portfela nie zjada alertu — mail idzie bez linii. +8 testów.

**Audyt 2026-08-05** (przed pierwszym realnym Must-Be-Won: rollover 4/5, próg roll-downu £9.14M):
- Stan zdrowy: 105 testów zielonych, kolektor w chmurze bez luk (29.07, 01.08, 02.08), watchdog OK, wszystkie ścieżki (`play` → `dashboard` → `backtest` → `roi_ledger`) przechodzą. Repo lokalne było 2 commity za `origin` — same dane, nie kod.
- ❌ **Błąd o jedno losowanie, jedna przyczyna w dwóch miejscach.** `next_draw_dates()` odpowiada „jaka jest następna data w kalendarzu", a oba wywołania potrzebowały „do którego losowania trafi kupon kupiony teraz". To rozjeżdża się w środy i soboty przed 19:30, czyli dokładnie wtedy, kiedy się z tego korzysta. (a) **`roi_ledger add`** w dniu losowania zapisywał kupony pod **następnym** losowaniem — kupione w sobotę MBW 08.08 wylądowałyby jako 08.12, a `settle` rozliczyłby je z wynikami losowania, w którym nigdy nie były. Cicho, bez błędu, na pierwszych realnych wpisach do ledgera. (b) **Prognoza MBW** spóźniała się o jedno losowanie: `make play` w środę 05.08 pokazywał `~2026-08-12` zamiast `~2026-08-08`.
- ✅ **Naprawione:** `upcoming_draw_date()` w `lottery/ev.py` — świadome zegara, nie tylko kalendarza (sprzedaż zamyka się **19:30 czasu UK**, strefa liczona jawnie przez `ZoneInfo("Europe/London")`, bo kolektor chodzi na UTC, a latem 19:00 UTC to już 20:00 w Londynie). `next_draw_dates()` zostaje jako czysta arytmetyka kalendarzowa z docstringiem odsyłającym do nowej funkcji. Ścieżka mailowa bez zmian w produkcji — odpala się zawsze po zamknięciu sprzedaży, jest osobny test pilnujący, że przebieg wieczorny i poranny retry nadal wskazują to samo losowanie. **Usunięty test, który utrwalał błąd** (`test_roi_ledger.py` asertował `Wed -> Sat` jako poprawne). +10 testów, razem 115.
- ❌ **Drugi błąd, w danych historycznych:** `backfill_prize_tiers.py:72` robi `re.sub(r"[^\d]", "", text)` — wycina litery, ale **skleja obie liczby**. Komórka „£10 Rolldown Prize: £24" daje `1024` zamiast 24, „£1 … £5" daje `15` zamiast 5. Dotyczy **93 losowań, 2017–2026 — wszystkich roll-downów**. Nie rusza żywej ścieżki EV (czyta `prize_tiers.csv` z fetchera, tam £24.00 poprawnie) ani kalibracji popularności (używa wyłącznie kolumny `winners`), ale psuje każdą historyczną analizę roll-downów — czyli dokładnie to, czym model roll-downu miałby być walidowany na czymś więcej niż jednym losowaniu.
- **Dlaczego mail nigdy nie przyszedł — nic nie jest zepsute.** Ostatnie losowanie Must-Be-Won to 3190 (18.07.2026, pula £9,559,451 → realnie +EV). Kolektor z alertami ruszył 19.07, **dzień później**. Od tego czasu 3191–3194 to zwykłe rollovery i SKIP jest poprawny. Okazje przychodzą ~10 razy w roku, system żyje 2,5 tygodnia.
- **Ile realnie warte jest czekanie.** ⚠️ Detektor „podbite niskie tiery" znajduje **93** losowania, ale to NIE są wszystko Must-Be-Won: 19 z nich to boosty promocyjne przy streaku 0–4 rolowań (mniej więcej co miesiąc, warte ~£0,37/linię — za mało, żeby przewrócić losowanie). Napędzanych capem jest **74**, z czego 4 (2017–18) mają pulę £0 = luki w danych. Właściwa populacja to **70 losowań, 2019-01-30 → 2026-07-18**: mediana puli **£12,18M**, przebija dzisiejszy próg £9,14M **68/70 = 97%**, tempo **9,4/rok**. Pudła tylko dwa, oba 2019 (£4,4M i £7,4M). Czyli obietnica z README (96%) trzyma się bez poprawek — a filtrowanie po samym boostcie niskich tierów zawyża liczbę okazji o jedną trzecią. Margines jest zdrowy: mediana leży 33% nad progiem, więc typowe MBW nie jest graniczne.

**Audyt 2026-08-06 — pierwszy mail +EV przyszedł i był ZŁY. Najdroższy błąd w projekcie.**
- **Co przyszło.** Losowanie 3195 (05.08) bez szóstki → rollover 5/5 → sobota **08.08 jest Must-Be-Won**, pula £8,391,429. Alert odpalił się pierwszy raz w historii i był **wierny co do cyfry**: jackpot, flaga, N = 6,574,552, EV +£0,038, próg £8,147,118 i wszystkie 5 linii odtwarzają się dokładnie z `build_portfolio(5, cond, seed=20260808)`. Pipeline działa — zadziałało też `upcoming_draw_date()` (pierwsza realna sobota po PR #10) i `calibrate_fixed_prizes` wzięło bazowe 10/1, nie roll-downowe 24/5.
- ❌ **Błąd: sprzedaż liczona ze zwykłych losowań, stosowana do Must-Be-Won.** Człon roll-downu to `J / N`, więc EV jest **malejące w N** — a `estimate_tickets_sold()` liczyło medianę z ostatnich losowań (prawie same zwykłe) i podstawiało ją do losowania MBW, które sprzedaje się **znacznie lepiej**. Zmierzone na 93 roll-downach z archiwum, każdy porównany z losowaniami tuż przed nim: **mediana 1,379x** (era 59 kul, n=89), kwartyle 1,07 / 1,69, stabilne 1,26–1,48 w blokach 2–4-letnich. Parowanie, nie przekrój, bo sprzedaż spada przez dekadę (przekrojowo wychodzi zawyżone 1,54x); okno 4 losowań, bo 1–2 przed roll-downem same są „gorące" (rollover 3–4) i zaniżają efekt do 1,11x.
- **Skala pomyłki na 08.08:** EV zerowało się już przy **6,781,862 liniach — margines 3,2%**. Do tego sama baza była zanieczyszczona: `prize_tiers.csv` ma 6 losowań, z czego 3190 to roll-down, czyli 1/6 próbki — mediana szła w górę o 10% (6,574,552 zamiast 5,913,930). Po poprawce: N = 8,161,222, **EV −£0,203**, próg £9,989,355 przy puli £8,39M — **brakuje £1,6M**. Werdykt: **SKIP**.
- **Dowód empiryczny, nie tylko model.** Losowanie 3190 (18.07) — jedyny MBW z pełnymi danymi, **zero szóstek w obu rundach**, czyli czysty roll-down (a boost roll-downu jest ślepy na popularność, więc niepopularna linia nic tu nie zyskuje). Wypłaciło £13,282,062 na ~8,38M linii = **£1,586 na linię za £2, czyli −£0,414**. Jego realna sprzedaż była o 27% wyższa niż to, co model podstawiał.
- ✅ **Naprawione:** `MBW_SALES_UPLIFT = 1,38` (+ kwartyle) w `lottery/ev.py` z pełną proweniencją; `estimate_tickets_sold(..., roll_down=)` **wycina roll-downy z bazy** (`rolldown_draw_numbers()`, sygnatura podbitego Match 3 — flaga `next_jackpot_roll_down` opisuje losowanie *po* wierszu, więc nie klasyfikuje najstarszego w pliku, czyli akurat 3190) i dopiero potem mnoży przez uplift. `next_draw_conditions(force_roll_down=)` — inaczej what-if `--roll-down` wyceniałby się po sprzedaży zwykłego losowania, czyli tym samym błędem. Nowe `sales_sensitivity()` liczy EV na kwartylach uplistu i daje flagę `robust`; advisor i **mail** pokazują teraz zakres i wprost mówią, czy werdykt przeżywa górny koniec sprzedaży. Nowy `scripts/calibrate_mbw_uplift.py` (raportuje, nie edytuje `ev.py` — jak `calibrate_popularity.py`). **+31 testów, razem 136**, w tym regresja na 08.08.
- ⚠️ **Konsekwencja dla tezy projektu — „97% MBW przebija próg" było liczone tą samą zaniżoną sprzedażą.** Przeliczone uczciwie: dla każdego historycznego MBW brana jest **jego własna zmierzona sprzedaż** i realna pula, wyceniane w dzisiejszych zasadach. Populacja odtwarza się dobrze (65 losowań napędzanych capem 2019-01-30 → 2026-07-18, **8,7/rok**, mediana puli £12,07M ≈ ustalone £12,18M). Wynik: **+EV jest tylko 27 z 65 = 42%**, mediana EV **−£0,139**, kwartyle −0,31 / +0,14. Czyli **~3,6 realnych okazji rocznie, nie ~8,7**. Ta sama próbka wyceniona po staremu (stałe N = 6–7,5M) dawała 91% i medianę +£0,78 — stąd brała się nadmierna obietnica. **Must-Be-Won to warunek konieczny, nie wystarczający**; decyduje werdykt advisora, nie kalendarz. Zastrzeżenie: to kontrfaktyk (stare losowania w dzisiejszych zasadach dwururundowych), więc traktować jako rząd wielkości, nie jako pomiar.

**Realizacja 2026-08-07 — cały plan ulepszeń wykonany w jeden dzień (PR #13–#18).**
Szczegóły, tabela PR-ów i **checklista na niedzielę 09.08** (scorecard MBW: uplift 1,27
na żywo, hipoteza −9% puli) w `plan-ulepszen-2026-08.md` na górze. Skrót: sprzedaż per
losowanie od 1994 + walidacja krzyżowa; day-aware N (sobota 1,27/środa 1,44 — werdykt
08.08 to **SKIP −£0,36 przy N=9,71M**); kolektor na JSON api-dfe z samonaprawą luk;
93 roll-downy przeparsowane; dokładna reguła roll-downu (EV-równoważna J/N) + dokładny
Kelly; model popularności zwalidowany 3 drogami (wagi zostają); wspólne ziarno
mail↔latest.json; typy MBW; ekran A&G; automatyczny scorecard po każdym MBW
(`data/mbw_validation.csv` + mail). Testy 143 → 194.

**Czeka na rozstrzygnięcie (stan 2026-08-05 22:30):**
- ~~**Pierwsze realne Must-Be-Won.** Rollover 4/5 po losowaniu 3194 (01.08).~~ **Rozstrzygnięte 2026-08-06:** 3195 bez szóstki → 08.08 jest Must-Be-Won z pulą £8,391,429, ale po naprawce sprzedaży to **SKIP (EV −£0,203)**, nie PLAY. Przewidywanie „pula £9–10M, tuż nad progiem" było zresztą optymistyczne w obie strony: pula wyszła niżej (£8,39M), a próg wyżej (£9,99M, nie £9,14M). Szczegóły w audycie 2026-08-06 wyżej.
- **Ledger nadal ma 0 kuponów** — ścieżka PLAY nigdy nie przeszła na prawdziwych pieniądzach. Sobota jest pierwszą okazją.
- ~~**93 losowania w `prize_tiers_history.csv` mają nadal zepsute `prize_per_winner`**~~ — **zrobione 2026-08-07**: `--redo-draws` na wszystkich 93, zero sklejonych wartości, roll-downowe Match 3 mają realne ceny £39–157. Odblokowana walidacja dokładnej reguły roll-downu (wyniki w `plan-ulepszen-2026-08.md` §3: M2=£5 w 61/88, człon M3 −9% systematycznie — hipoteza: ogłoszona pula > realnie rozdzielona).
- **Realne opóźnienia GitHub Actions** (z historii przebiegów, ważne przy planowaniu): cron wieczorny 21:45 UTC odpala się ~55 min później, poranny 06:00 UTC ~2,5 h później, watchdog 12:00 UTC ~1–1,5 h później. Czyli kolekcja po losowaniu realnie ~23:40 BST, nie 22:45.

**Checklista po losowaniu** (`git pull` zawsze najpierw — kolektor commituje dane z chmury):
```bash
git pull                                             # dane z chmury
tail -3 data/merged_lottery_data.csv                 # czy losowanie zebrane
make play                                            # werdykt + rollover 4/5 -> ?
gh run list --workflow=collect.yml --limit 2         # czy kolekcja przeszla
gh run view --log | grep ev-alert                    # SKIP czy PLAY (bez czekania na maila)
make roi                                             # settle + ROI (jesli grane)
```
Brak maila sam w sobie nic nie znaczy — `[ev-alert] SKIP (EV £-x.xx)` w logu to poprawne zachowanie. Dopiero SKIP **przy** Must-Be-Won z pulą nad progiem = błąd do zdiagnozowania.

**Kamienie milowe:**
- [ ] **+1 miesiąc** (~9 losowań): sprawdzić stabilność estymatora N. ~~5+bonus prize~~ zrobione (£1M). ~~Match 3/Match 2~~ zrobione 2026-07-28 (£10/£1, roll-down zdemaskowany). ~~Mapowanie tierów~~ — zrobione (audyt 2026-07-21). ⚠️ **N okazało się najwrażliwszym parametrem w całym modelu** (audyt 2026-08-06) — przy roll-downie EV to w praktyce `J/N`, więc to jego stabilność decyduje o wszystkim, nie ceny tierów.
- [ ] **Walidacja upliftu 1,38 na żywych danych.** Stała jest zmierzona na archiwum (n=89, kwartyle 1,07/1,69), ale ani jedno losowanie Must-Be-Won nie przeszło jeszcze przez `prize_tiers.csv` z *poprawnym* estymatorem po obu stronach. Po każdym kolejnym MBW: porównać zmierzone N z tym, co model przewidział (`scripts/calibrate_mbw_uplift.py`), i sprawdzić, czy 1,38 nie dryfuje. Przy 3–4 obserwacjach warto przeliczyć.
- [ ] **Rozdzielić „Must-Be-Won" od „duża pula".** Roll-down zawsze niesie największą pulę swojego cyklu, więc zmierzone 1,38 miesza oba efekty. Dla puli poniżej mediany (jak £8,39M na 08.08) sprzedaż może być bliżej dolnego kwartyla, czyli model jest wtedy zbyt pesymistyczny. Wymaga pul per losowanie — **zablokowane przez zepsute `prize_per_winner`** w 93 roll-downach archiwum (`backfill_prize_tiers.py --redo-draws`).
- [x] ~~**Ujednolicić linie z maila i z `latest.json`.**~~ — **zrobione 2026-08-07**: jedno `default_portfolio_seed(draw_date)` w `lottery/ev.py`, używane przez obie ścieżki (`ev_play.py` bez `--seed` i `ev_alert.py`). Mail i `roi_ledger add --from-latest` proponują te same kupony; jawny `--seed` nadal wygrywa.
- [x] ~~**Roll-down watch:** pierwsze losowanie Must-Be-Won~~ — zrobione na 3190 (patrz wyżej), model potwierdzony w granicach 2%.
- [x] ~~**Prognoza MBW**~~ — zrobione 2026-07-28: `forecast_must_be_won()` + `ROLLOVER_CAP = 5` (wyprowadzony z danych). Advisor i dashboard pokazują „rollover 2/5 — Must-Be-Won za 4 losowania, ~2026-08-08, jeśli nikt wcześniej nie trafi". To górne ograniczenie czasu oczekiwania, nie harmonogram: trafiony jackpot zeruje licznik. Wiążący sygnał na jedno losowanie do przodu pozostaje flaga `next_jackpot_roll_down` z oficjalnego feedu.
- [x] ~~**Sprzątanie martwego kodu**~~ — 2026-07-28 usunięte `scripts/utils/{cross_validation,memory_monitor,model_utils,utils,setup_logging}.py` + `_backup/` (**1408 linii**, nic ich nie importowało; `cross_validation` sięgał po skasowany w Fazie 1 `scripts.train_models`). Przy okazji zniknęło ryzyko przesłonięcia funkcji `setup_logging` przez moduł o tej samej nazwie, opisane w komentarzach w dwóch plikach. Kalendarz losowań (śr./sob.) był zduplikowany w `roi_ledger.py` — teraz jedno `next_draw_dates()` w `lottery/ev.py`.
- [ ] **Ledger ≥ 20 zagranych kuponów** (tylko przy PLAY!) → pierwszy raport ROI z sensowną próbką.

**Kryterium „zaczynamy grać":** to nie „wystarczająco danych", tylko werdykt advisora — graj wyłącznie gdy `make play` mówi **PLAY** (duży rollover / Must-Be-Won), skalibrowanym modelem, z budżetem (np. max £10/losowanie) i każdym kuponem w ledgerze. **Od 2026-08-06 dochodzi drugi warunek: `Holds across range: YES`.** Sam PLAY nie wystarczy — werdykt, który przechodzi tylko na centralnej estymacie sprzedaży, nie jest okazją, tylko zaokrągleniem (tak wyglądało 08.08: +£0,038 przy marginesie 3,2%). Must-Be-Won to warunek konieczny, nie wystarczający — realnie +EV jest ~42% takich losowań, nie 97%. Uczciwie: „ztiuningowany model" w tym projekcie znaczy *dokładniejsze EV*, nigdy przewidywanie liczb — tego nie umożliwią żadne dane.

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
