# Plan ulepszeń — „world-best" toolkit EV dla UK Lotto (2026-08-07)

Research z 2026-08-07: trzy niezależne kwerendy (literatura naukowa o +EV w loteriach,
źródła danych UK Lotto zweryfikowane żywymi fetchami, przegląd ML-owych repo na GitHubie).
Wszystkie URL-e w tym pliku były sprawdzone tego dnia. Uzupełnia `plan.md` (FAZA 7);
niczego z niego nie unieważnia.

---

## 2026-08-08 — wheel generator (side-play na czas oczekiwania na maile MBW)

Jednorazowy generator skróconego wheela na puli 12 najmniej granych numerów —
zabawka „w międzyczasie", dopóki scorecardy MBW nie zaczną przychodzić; NIE zmienia
EV i nie dotyka `latest.json` ani ledgera.

- `scripts/wheel_play.py` — greedy covering design celujący w „3 if 4"; gwarancje
  **mierzone wyczerpująco** na wygenerowanych kuponach, nie przepisane z tabelki.
  Pula = pasmo remisowe najniższej wagi popularności, rozrzucone (nie ciągłe),
  żeby filtr 3-kolejnych miał kandydatów. Domyślnie 10 linii / pula 12.
- `scripts/backtest_wheel.py` — backtest na całej erze 6/59 (1 147 losowań,
  realne nagrody per losowanie z `prize_tiers_history.csv`). Wyniki 2026-08-08:
  - gwarancja „3 if 4" trzyma w 14/14 przypadków t≥4, zero naruszeń;
  - 136 trafień M3+ vs 124,6 oczekiwanych (85. percentyl randomu = fart, nie edge);
    zwrot 21,3% vs ~20% mediany losowych portfeli — **EV identyczne z randomem**;
  - clumping realny: wariancja wygranych/losowanie 0,22 vs 0,11 u randomu z 59;
    wygrane rzadziej (87 vs 118 losowań), ale seriami po 2–4 kupony;
  - χ²=38,1 (krytyczne 76,8) i test hot/cold (k=5/10/20) → losowania jednostajne,
    **żadna metoda prawdopodobieństwa nie poprawi trafień** — potwierdza §8;
  - jedyna gałka: `--pool-size` (12 → t≥4 ~1,3×/rok; 14 → ~2,5×/rok; tabela w skrypcie).

Testy: 194 → 207 zielone (13 nowych). Werdykt bez zmian: wheel to rozrywka o tym
samym EV co quick pick, z ładniejszym kształtem wygranych; prawdziwa gra zostaje
w `ev_play.py` i czeka na MBW.

## STATUS 2026-08-07 wieczorem — CAŁY PLAN WYKONANY (PR #13–#18, jeden dzień)

| PR | Co weszło |
|---|---|
| #13 | sprzedaż per losowanie od 1994 (Merseyworld) + walidacja krzyżowa dwóch estymatorów N (zgodność 5%) |
| #14 | day-aware sprzedaż: baza tego samego dnia tygodnia, uplift sobota 1,27 / środa 1,44; werdykt 08.08 → SKIP −£0,36 przy N=9,71M |
| #15 | kolektor JSON (api-dfe) jako główny tor; 93 roll-downy przeparsowane; dokładna reguła podziału (EV-równoważna J/N → weszła do rozkładu, nie średniej); dokładny Kelly (uczciwy wynik: grosze przy detalicznym bankrollu) |
| #16 | model popularności zwalidowany 3 drogami (spójność między tierami, ogon współzwycięzców χ²=9,0, OOS-remis → wagi zostają); kolektor sam leczy luki (okno 180 dni) |
| #17 | wspólne ziarno portfela mail↔latest.json (realny bug przed pierwszym PLAY); typy MBW cap-driven/special-event; ekran Abrams–Garibaldi (2. opinia dla zwykłych losowań); `--ordinary` |
| #18 | automatyczny scorecard po każdym MBW (N vs prognoza, pula ogłoszona vs rozdzielona) → `data/mbw_validation.csv` + mail |

Testy: 143 → **194 zielone**. Próba generalna kolektora w chmurze przeszła 07.08 ~17:45 UTC
(JSON + scorecard + `[ev-alert] SKIP (EV £-0.36)`).

### CHECKLISTA — niedziela 2026-08-09 (albo sobota po ~23:45)

1. `git pull` — kolektor commituje dane losowania 3196 ~23:40 BST (cron chodzi ~1 h
   późno; niedzielny retry 06:00 UTC ~2,5 h późno domyka awarie).
2. **Mail „LOTTO MBW scorecard: draw 3196"** (też w logu collect.yml; pierwszy wiersz
   w `data/mbw_validation.csv`). Jak czytać:
   - `uplift_measured` vs 1,27 — pierwsza żywa walidacja sobotniego uplistu;
     w widełkach 1,19–1,31 (kwartyle) = stała trzyma.
   - `pool_ratio` — test hipotezy −9% (ogłoszona pula > realnie rozdzielona).
     Jeśli na żywo też ~0,91 → człon J/N jest ~9% optymistyczny i `cond.jackpot`
     zasługuje na haircut w `line_ev` — **to jest następna zmiana modelu**.
   - `n_error` — prognoza N vs pomiar; margines SKIP to £3,3M w puli, więc tylko
     gigantyczna pomyłka odwróciłaby werdykt wstecznie.
3. Maila PLAY **ma nie być** (SKIP poprawny). Jeśli przyjdzie — bug do diagnozy.
4. Jeśli ktoś trafił szóstkę (~35% szans) — roll-downu nie było, scorecard poprawnie
   milczy, licznik wraca do 0, następne MBW za ~5–6 tygodni.
5. Przy 3–4 wierszach w `mbw_validation.csv` (kilka miesięcy): przeliczyć
   `scripts/calibrate_mbw_uplift.py` i zrewidować 1,27/1,44.

Poza checklistą jedyna odłożona rzecz z „miną": model klastrowy B&M 2011 — wróci sam,
gdy test ogona w `calibrate_popularity.py` krzyknie „TAIL DIVERGES".

### Testy e2e całej aplikacji — 2026-08-07 wieczorem ✅

Przetestowana każda funkcjonalność na realnych danych, po kolei:

| # | Ścieżka | Wynik |
|---|---|---|
| 1 | `make test` | 194/194 zielone |
| 2 | `make play` na żywo | SKIP −£0,36, N=9,71M (sobota ×1,27), spójne z chmurą |
| 3 | What-ify: `--jackpot --roll-down --ordinary --seed --lines --bankroll` | wszystkie działają; Kelly, ekran A&G i portfel renderują się poprawnie |
| 4 | Ochrona `latest.json` | what-if **nie nadpisuje** (md5 przed/po identyczne) |
| 5 | `make dashboard` | 25 kafli, „SKIP" + „Must-Be-Won (cap-driven)" widoczne |
| 6 | `make backtest` (186 losowań) | uczciwe „no edge over random" na wszystkich metodach (p=0,23–0,53) |
| 7 | Ledger pełny cykl: `add` → `settle` → `report` (piaskownica na realnych wynikach 3190, potem sprzątnięte) | 2 linie rozliczone poprawnie z obu rund, ROI −100% (1 trafienie/linia = brak nagrody — zgodnie z tabelą) |
| 8 | `ev_alert` | `SKIP (EV £-0.36) - no alert sent` — bez fałszywego maila |
| 9 | `post_mbw_validation` | poprawnie milczy (3195 nie było roll-downem) |
| 10 | Watchdog | OK — 3195 obecne, 72 wiersze tier |
| 11 | `make nightly` | przechodzi, zapisuje best_ensemble.json |
| 12 | `make sales` + walidacja | 3 195 losowań, mediana zgodności 1,052 — odtwarza się |
| 13 | `calibrate_mbw_uplift` | sekcja day-aware odtwarza zainstalowane stałe (Sat 1,270, Wed 1,440) |
| 14 | `calibrate_popularity` + 3 walidacje | χ²=9,0 adekwatny, OOS installed 0,509 — odtwarza się |
| 15 | Legacy `./predict_tonight.sh` | działa (10 linii ensemble; trzymane wyłącznie jako sanity-check — backtest wyżej mówi „no edge") |
| 16 | `post_draw.sh` — pełna rutyna lokalna | wszystkie kroki po kolei: fetch (JSON) → scorecard → settle → dashboard → werdykt → alert; zero błędów |
| 17 | Kolektor w chmurze (dress rehearsal, run 31203761207) | JSON + scorecard + `[ev-alert] SKIP` — przeszedł w Actions ~17:45 UTC |
| 18 | Higiena repo po testach | `git status` czysty — artefakty testów w gitignore albo idempotentne |

**Znaleziska (nic zepsutego, dwie obserwacje):**
1. `--force` bez argumentów what-if **zapisuje wymuszony portfel do `latest.json` w dzień
   SKIP** — więc `roi_ledger add --from-latest` po `--force` zapisałby linie losowania,
   którego advisor nie poleca. To zachowanie zamierzone (`--force` = „gram mimo wszystko"),
   ale warto pamiętać: po zabawie z `--force` odpal zwykłe `make play`, żeby przywrócić
   prawdziwy werdykt do pliku.
2. Sekcja day-aware w `calibrate_mbw_uplift.py` widzi 27 środowych roll-downów, wcześniejsza
   analiza 31 — różnica z metody detekcji (moda vs mediana bazy Match 3). Stałe wychodzą
   identyczne w granicach zaokrąglenia, więc bez zmian; odnotowane, gdyby przy przyszłej
   rekalibracji liczby się nie zgadzały.

---

## 0. Werdykt z researchu — gdzie jesteśmy naprawdę

**Ten projekt już robi dokładnie te dwie rzeczy, które według recenzowanej literatury
w ogóle działają** przy loteriach: wycenę EV z modelem popularności (Haigh 1997; Baker
& McHale 2009) i wycenę roll-downów (mechanizm Cash WinFall). Nic opublikowanego dla
6/59 nie jest dalej niż my — nieformalna analiza Buntinga (2020) zakłada brak dzielenia
nagród i stałą sprzedaż, czyli błędy, które u nas są już naprawione.

**Czego na pewno NIE robić: predykcji liczb.** Przegląd najpopularniejszych repo
(yangboz/LotteryPrediction 290★, KittenCN 269★, CorvusCodex 149★ i inne): **ani jedno
nie pokazuje żadnego backtestu**, a jedyny znaleziony rzetelny publiczny backtest LSTM
(70 lat niemieckiego 6/49) **przegrał z losowym wyborem** (0.7352 vs 0.7359 trafień
średnio; [Medium/Jonas David](https://medium.com/mind-code/statistical-deception-predicting-lottery-numbers-with-ai-d555b521e5a5)).
Testy χ² na UK Lotto: 2 231 losowań zgodnych z pełną losowością
([Data in Brief 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC5536828/); Haigh 1997,
JRSS-A; Joe 1993; Genest 2002). Nasz własny backtest mówi to samo. „World-best" ≠
lepsza predykcja — to **dokładniejsze EV, lepsza prognoza sprzedaży i lepszy model
popularności**. Cała reszta tego planu to te trzy osie.

---

## 1. PRIORYTET 0 — Dane: sprzedaż per losowanie z Merseyworld 📈

**Największe pojedyncze odkrycie researchu.** N (liczba sprzedanych linii) to najwrażliwszy
parametr modelu (audyt 2026-08-06: EV roll-downu ≈ J/N), a my estymujemy je pośrednio
z liczby zwycięzców i mnożymy przez płaski uplift 1,38. Tymczasem
**lottery.merseyworld.com publikuje sprzedaż per losowanie dla wszystkich ~3 195 losowań
od 19.11.1994**, odświeżane codziennie ~1:00:

- CSV sprzedaży: `https://lottery.merseyworld.com/cgi-bin/lottery?sales=1&year=2026&display=CSV`
  (pola `Day,DD,MMM,YYYY,Sales,%Chg` w £; `sales=1` sobota, `sales=2` środa; `year=0` = wszystkie lata).
  Zweryfikowane: śr. 05.08.2026 = £10 523 874, sob. 01.08 = £17 072 084.
- Strony archiwum per losowanie: `https://lottery.merseyworld.com/archive/Lott{N}.html`
  (N bez zer wiodących, od ~100 do bieżącego 3195) — sprzedaż, pełne rozbicie nagród
  per runda, księgowość rolloveru (kwota i % puli), maszyna/zestaw kul.
  Najbogatsze pojedyncze źródło per-draw, jakie istnieje.
- ⚠️ Zastrzeżenie: od 12.07.2003 operator nie publikuje sprzedaży per losowanie —
  liczby Merseyworld po tej dacie to **estymaty wyprowadzone z % funduszu nagród**.
  Nadal złoto: niezależne od naszego estymatora z winner-counts, więc mamy **dwa
  niezależne estymatory N** do wzajemnej walidacji.

**Zadania:**
- [ ] `scripts/fetch_sales.py` — scraper obu źródeł do `data/sales_history.csv`
  (draw_number, date, sales_gbp, source). `backfill_history.py` już używa
  Merseyworld cgi-bin, więc wzorzec requestów jest gotowy.
- [ ] Walidacja krzyżowa: `sales/2` (linia £2, uwaga na erę £1 przed 10.2013) vs
  `estimate_tickets_sold()` na 1 126 losowaniach z winner-counts. Raport rozbieżności;
  jeśli zgodność dobra → Merseyworld zostaje głównym źródłem N historycznego.
- [ ] Przeliczyć uplift MBW na realnych sprzedażach (koniec sporu „1,38 z parowania
  okien”) i **rozdzielić „Must-Be-Won" od „duża pula"** — kamień milowy z plan.md,
  dotąd zablokowany, odblokowuje się bez czekania na `--redo-draws`.

### Wyniki §1 — wykonane 2026-08-07 ✅

Scraper działa (`scripts/fetch_sales.py`, `make sales`): **3 195 losowań** sprzedaży
w `data/sales_history.csv` (19.11.1994 → dziś). Korekty względem researchu:
`sales=2` to scratchcards, seria `sales=1` niesie oba dni; strony per-rok dublują
wiersze. Co mówią dane:

- **Walidacja krzyżowa** (1 130 losowań z oboma źródłami): mediana
  MW/winner-count = **1,052** — dwa niezależne estymatory N w 5% od siebie.
  MW **nie zawyża roll-downów**: na nich mediana 0,995 vs 1,058 na zwykłych
  (34% figur roll-downowych to okrągłe estymaty vs 14% — więcej szumu, zero biasu
  w górę). Rozjazd na 3190 (×1,27) to outlier.
- **Uplift potwierdzony niezależnie**: mediana z realnych sprzedaży (baza = mediana
  4 poprzednich, jak w `ev.py`) = **1,391** vs 1,379 z winner-counts. Kwartyle
  węższe: 1,13 / 1,55 (było 1,07 / 1,69).
- **Hipoteza „mała pula → niższy uplift" OBALONA**: regresja log(uplift) ~ log(pula)
  daje współczynnik ≈ 0 (−0,016; małe pule mają wręcz wyższy uplift 1,46 vs 1,31).
  Zastrzeżenie z audytu 2026-08-06 o 08.08 można zamknąć — pesymizm modelu nie
  wynikał z puli.
- ⚠️ **Prawdziwy brakujący regresor to DZIEŃ TYGODNIA.** Sobota sprzedaje 1,59×
  środy (zwykłe losowania, ostatnie ~50). Uplift MBW liczony do bazy z TEGO SAMEGO
  dnia: **sobota 1,169 (q25 1,116 / q75 1,297; od 2023 stabilnie 1,173, n=27) —
  środa 1,389 (1,215 / 1,531; od 2023: 1,57, ale n=9)**. Stała 1,38 na mieszanej
  bazie działa dla sobót tylko dlatego, że 62/93 kalibracji to soboty i artefakty
  się znoszą; dla **środowego MBW zawyża N o ~25–30% → EV zbyt pesymistyczne →
  ryzyko przegapienia +EV** (bezpieczny kierunek błędu, ale kosztowny: okazji jest
  ~4/rok).
- ✅ **Day-aware sprzedaż wdrożona 2026-08-07** (drugi PR): `estimate_tickets_sold`
  z bazą tego samego dnia tygodnia + `mbw_uplift(draw_date)` per dzień
  (**sobota 1,27 q 1,19/1,31 — środa 1,44 q 1,38/1,54**, w definicji zgodnej
  z oknem estymatora: 20 losowań, tylko zwykłe, ten sam dzień),
  `DrawConditions.draw_date`, `sales_sensitivity` z kwartylami per dzień (dużo
  węższe niż mieszane 1,07/1,69 → flaga `robust` przestaje rozciągać się na
  różnice między dniami), `calibrate_mbw_uplift.py` raportuje sekcję day-aware
  z realnych sprzedaży. Werdykt na 08.08 po zmianie: **SKIP, EV −£0,36 przy
  N=9,71M** (baza sobotnia; poprzednio −£0,20 przy 8,16M — mieszana baza
  zaniżała sobotni poziom). Regresja absolutna (R²=0,89 in-sample) świadomie
  NIE weszła do prognozy — liniowy trend nie łapie dekady spadku sprzedaży,
  formuła względna (uplift × baza) zostaje.

## 2. PRIORYTET 0b — Naprawić żywy feed oficjalny 🔧

Zweryfikowane żywcem 2026-08-07:

- Stary `https://www.national-lottery.co.uk/results/lotto/draw-history/csv` **już nie
  istnieje** — 308 na `/draw-history/xml` (tylko ostatnie losowanie). Nasz fetcher to
  obsługuje (`_ingest_official_xml`), ale `data/.download_state.json` wciąż woła stary
  URL i polega na przekierowaniu. Przepiąć na docelowe endpointy.
- **Właściwe API to `https://api-dfe.national-lottery.co.uk`** (bez auth; wymaga
  przeglądarkowego User-Agent, inaczej 403 z AWS WAF):
  - `GET /draw-game/results/1/download` — CSV (DrawDate, Ball1..6, Bonus, Ball Set,
    Machine, DrawNumber), tylko ~180 dni, tylko runda 1.
  - `GET /draw-game/results/6/{drawNo}` i `/draw-game/results/1/latest` — **pełny JSON
    per losowanie**: obie rundy, `prizeLevels[]` z `drawRound`, `allWinnersCount`,
    `prize.prizeCents`, `prizeFundCents`, `prizeCap`, `prizeRollDown` — czystsze niż
    parsowanie XML i z polami, których XML nie ma (fundusz per tier!).
  - Okno tylko ~180 dni (najstarszy działający drawNo 3145) → kolektor commitujący do
    gita pozostaje jedyną pełną historią; to potwierdza architekturę, nie zmienia jej.
- [x] **Zrobione 2026-08-07:** `_ingest_official_json` jest głównym torem kolektora
  (retry, potem pełny fallback na stary tor XML/CSV). Nowe kolumny w
  `prize_tiers.csv`: `prize_per_winner` (wprost z API) i `tier_roll_down`
  (oficjalny marker per tier — koniec heurystyki podbitego Match 3 dla nowych
  losowań). Pola wyprzedzające (`next_jackpot_estimate/roll_down`) dociągane
  z XML best-effort; gdy XML martwy, flaga MBW wyprowadzana z capu rolloverów
  (5) — alert przeżyje śmierć przekierowania. Zweryfikowane na żywo na 3195.
- [x] **Lepiej niż watchdog z drugim źródłem — kolektor sam się leczy (2026-08-07):**
  `recover_missing_draws()` dociąga każdą lukę między ostatnim zebranym a
  najnowszym losowaniem z okna ~180 dni JSON API (per numer,
  `/draw-game/results/6/{n}`), zanim zingestuje najnowsze. Przegapione okno
  przestało być trwałą stratą; alert watchdoga znaczy teraz „padły oba
  przebiegi", nie „straciliśmy dane". Scraper archiwum zostaje trzecią linią.

## 3. PRIORYTET 1 — Dokładna reguła roll-downu zamiast J/N 🎯

Z reguł gry (potwierdzone przez lottery.co.uk/lotto/must-be-won-draws i komunikat TNL):
roll-down **najpierw płaci £5 każdemu zwycięzcy Match 2, a dopiero resztę rozdziela
między zwycięzców Match 3** — nie proporcjonalnie po wszystkich tierach i nie jednym
strumieniem J/N. Nasze przybliżenie zwalidowało się na 3190 w 2% (J/N = £1,28 vs £1,26),
ale dokładna reguła daje:

**Wykonane 2026-08-07, z jedną ważną korektą teoretyczną:**
- [x] **Podział jest EV-równoważny J/N** — każda reguła rozdająca całą pulę
  zwycięzcom niższych tierów daje losowemu kuponowi J/N w oczekiwaniu (stąd
  walidacja 98% na 3190). Dokładna reguła zmienia ROZKŁAD, nie średnią — weszła
  więc nie do `line_ev` (tam J/N zostaje), ale do `rolldown_tier_boosts()` +
  `line_return_distribution()` (mieszanina: z P(nikt nie trafi 6) ≈ 65% tiery
  z boostem, inaczej bazowe), na których stoi Kelly (§6).
- [x] **93 roll-downy przeparsowane** (`--redo-draws`, 0 sklejonych wartości;
  M3 per-winner £39–157). Walidacja reguły na 88 post-2018: **M2 = dokładnie £5
  w 61/88**, człon M3 odtwarza się z medianowym błędem **−9%, systematycznym**.
  ⚠️ Hipoteza: kolumna `Jackpot` archiwum to ogłoszony estymat, a realnie
  rozdzielana pula biega ~9% niżej — jeśli tak, człon J/N liczony na ogłoszonej
  puli jest ~9% optymistyczny. Outlier: 3131 (specjalne świąteczne MBW
  24.12.2025) ma w archiwum ewidentnie błędną pulę.
  **Zautomatyzowane 2026-08-07:** `scripts/monitoring/post_mbw_validation.py`
  odpala się po każdej kolekcji (collect.yml + post_draw.sh); po roll-downie
  liczy zmierzone N vs prognozę (żywy scorecard upliftu 1,27/1,44), ogłoszoną
  pulę vs realnie rozdzieloną (rozstrzyga hipotezę −9%), akumuluje wyniki w
  `data/mbw_validation.csv` (commitowane z danymi) i wysyła mail. Pierwszy
  wpis pojawi się sam po sobocie 08.08.
- [x] **Dashboard i advisor odróżniają typy MBW (2026-08-07):** `mbw_type()` —
  „cap-driven" przy liczniku ≥ 5, „special-event" gdy flaga roll-down przyszła
  bez pełnego licznika (świąteczne ~£15M, jak 3131). Advisor, mail i dashboard
  nazywają typ; nowy `--ordinary` w `ev_play` pozwala też zapytać o zwykłe
  losowanie, gdy feed flaguje roll-down (dotąd nie dało się).

## 4. PRIORYTET 1 — Endogeniczny model sprzedaży zamiast stałej 1,38 📊

Literatura (UK-specyficzna!): sprzedaż to funkcja ogłoszonego jackpotu, rolloveru i
momentów rozkładu wypłat — [Walker & Young 2001, Economic Journal](https://www.blackwellpublishing.com/specialarticles/Ecoj666.pdf)
(gracze lubią średnią i skośność, nie lubią wariancji); Forrest, Gulley & Simmons 2000
(elastyczność ceny efektywnej ≈ −1; sprzedaż UK jest prognozowalna na poziomie
pojedynczego losowania); Cook & Clotfelter 1993 (skala jackpotu → sprzedaż).

- [ ] Z danymi Merseyworld (§1): regresja `N ~ f(ogłoszony jackpot, rollover_count,
  dzień tygodnia, trend, MBW-flag)` na całej erze 59 kul. Zastępuje płaski uplift
  1,38 predykcją z przedziałem — `sales_sensitivity()` dostaje przedział z modelu
  zamiast kwartyli historycznych.
- [ ] Wtedy rozstrzyga się hipoteza z audytu 2026-08-06: mała pula MBW (£8,39M)
  sprzedaje się bliżej dolnego kwartyla → być może 08.08 był mniej SKIP, niż mówił
  model. Po każdym MBW: porównanie prognozy N z pomiarem (kamień milowy z plan.md).

## 5. PRIORYTET 1 — Model popularności klasy Baker & McHale 🧮

Nasz model: niezależne wagi per liczba × mnożniki wzorców, kalibrowane na Match-3
winner counts (fit forward-modelu). Literatura mówi, jak zrobić to lepiej — na danych,
które już mamy (1 126+ losowań winner counts per tier):

- [Baker & McHale 2009, JRSS-A 172(4)](https://academic.oup.com/jrsssa/article/172/4/813/7084574):
  conscious selection ⇒ **silna naddyspersja** liczby zwycięzców i **korelacje między
  tierami w obrębie nocy** (popularność wylosowanego zestawu pcha M3/M4/M5 naraz).
  Wniosek praktyczny: (a) kalibracja z likelihoodem ujemno-dwumianowym, nie
  poissonowskim; (b) używać **łącznie** M2+M3+M4 danej nocy do inferencji popularności
  zestawu — ostrzejsze wagi z tych samych danych.
- [Baker & McHale 2011, JRSS-A 174(4)](https://academic.oup.com/jrsssa/article/174/4/1071/7077904):
  3-parametrowy **model preferencji kombinacji/klastrów** (gracze wybierają klastry
  podobnych kuponów — wzory na playslipie, daty) — modele per liczba **z dowodem** nie
  odtwarzają obserwowanych korelacji winner-counts. Simon 1998 (realne dane Camelot,
  69,2M kuponów jednego losowania): ~2 000 kombinacji miałoby **>200 współzwycięzców
  jackpotu** — ogony są dużo cięższe niż w modelu niezależnych wag.
- [Stern & Cover 1989, JASA](https://isl.stanford.edu/~cover/papers/paper91.pdf):
  jeśli zostajemy przy wagach per liczba — rozkład po kombinacjach powinien być
  maksymalno-entropijny przy zadanych marginesach (zasada, nie heurystyka).

**Wykonane 2026-08-07 — wynik: walidacja zamiast podmiany.** Trzy nowe sekcje
w `calibrate_popularity.py` (uruchamiają się przy każdej pełnej kalibracji):

- [x] **Spójność między tierami** — fit per tier z un-dampingiem 6/k daje
  M4: 1,228/1,094/0,838 · M3: 1,230/1,100/0,834 · M2: 1,205/1,092/0,850 —
  trzy niezależne stopnie dopasowania odzyskują te same wagi. To odpowiedź na
  zarzut B&M o modele per liczba: na naszej precyzji model JEST adekwatny.
  Naiwny łączny fit ważony wariancją **przegrał out-of-sample** z M3-only
  (przeważa M2, najsłabszy sygnał na jednostkę szumu) — świadomie nie wszedł.
- [x] **Test ogona współzwycięzców** (bezpośredni test `expected_cowinner_share`):
  obserwowane 963/158/21/3/2 zwycięzców jackpotu (0/1/2/3/4+) na 1 147
  rundach vs model 942/179/22/2,7/0,4 — χ²=9,0, model adekwatny. Jedyny ślad
  klastrów Simona: 2 losowania z 4+ zwycięzcami vs 0,4 oczekiwane — poniżej
  istotności; klastrowa korekta B&M 2011 jest poniżej progu wykrywalności
  przy naszej wielkości próby. Wagi 1,23/1,10/0,83 **zostają** (remis OOS
  z refitem: corr 0,509 vs 0,507).
- [ ] Model klastrowy B&M 2011 — wrócić, jeśli ogon 4+ zacznie się powtarzać
  (test χ² w raporcie krzyknie „TAIL DIVERGES").
- [ ] Do werdyktu dodać **rozkład dzielenia**, nie tylko średnią: P(jackpot dzielony),
  kwantyle zwrotu linii (Matheson & Grote: różnica 1% vs 11% okazji to wyłącznie
  człon dzielenia — [In Search of a Fair Bet](https://web.williams.edu/Economics/wp/mathesonlottery.pdf)).

## 6. PRIORYTET 2 — Kelly per tier: z PLAY/SKIP do „ile grać" 💰

[MacLean, Ziemba & Blazenko 1992](https://link.springer.com/article/10.1023/A:1018969727211):
przy przewadze siedzącej w jackpocie (p ~ 10⁻⁷) pełny Kelly to ~65 kuponów po $1 na
**$10M majątku** — edge jackpotowy jest praktycznie niegrywalny. Ale **edge roll-downu
siedzi w Match 3 (p ≈ 1/97) i Match 2 (p ≈ 1/10)** — Kelly przy takich
prawdopodobieństwach jest o rzędy wielkości większy; dokładnie dlatego syndykat MIT
mógł racjonalnie stawiać sześciocyfrowe kwoty na Cash WinFall
([raport Inspektora Generalnego MA, 2012](https://www.mass.gov/files/documents/2016/08/vv/lottery-cash-winfall-letter-july-2012.pdf)).

- [x] **Zrobione 2026-08-07** — `kelly_stake()` z DOKŁADNĄ maksymalizacją
  E[log(1+f·r)] (bisekcja pochodnej; przybliżenie f\*≈E[r]/E[r²] jest dla wypłat
  loteryjnych błędne o rzędy wielkości — dodatni gruby ogon pompuje E[r²], choć
  strata jest ograniczona do ceny kuponu). Advisor i mail drukują stawkę.
  **Uczciwy wynik: pełny Kelly dla pojedynczej linii to f\* ≈ 3e-6** — nawet
  +40% edge na MBW uzasadnia grosze przy detalicznym bankrollu, dokładnie jak
  u MacLean-Ziemba (83% edge → 65 biletów/$10M). Komunikat mówi to wprost:
  „edge jest realny, ale każda linia to stawka rozrywkowa, nie wzrostowa".
  To zamyka spór „ile grać": budżet £10–20/MBW jest w pełni racjonalny jako
  rozrywka z dodatnim EV, nie jako inwestycja.
- [x] **Ekran A&G wdrożony (2026-08-07)** — `abrams_garibaldi_screen()`, druga
  opinia TYLKO dla zwykłych losowań (ich twierdzenia wyceniają pari-mutuel
  jackpot, nie roll-down — dla MBW zwraca None zamiast się mylić). Jest
  celowo surowszy niż nasz dokładny próg: ich warunek to +EV **odporne na
  dowolną sprzedaż** (dla UK ≈ £200M puli — oba warunki naraz), gdy nasz
  break-even £30,2M zakłada bieżące N. Uczciwy wniosek z ekranu: rekord ery
  to £52,9M, więc **żadne zwykłe losowanie UK Lotto nigdy nie było odpornie
  dobrym zakładem** — cała przewaga żyje w roll-downach, zgodnie z naszym
  modelem. Zastrzeżenie: ich twierdzenie zakłada F≥0,8, UK ma ~0,64 — progi
  są orientacyjne, co dla drugiej opinii wystarcza.
  [Finding good bets in the lottery](https://arxiv.org/abs/2507.01993).

## 7. PRIORYTET 2 — Higiena i operacje 🧹

- [ ] **Usunąć ścieżkę LSTM na dobre** (`models/checkpoints/*.h5/pkl`, legacy
  `new_predict.py`): research jednostronny (patrz §0), a `README` już mówi prawdę.
  Backtest z p-value zostaje — to nasz dowód uczciwości, nie narzędzie predykcji.
- [ ] **Ujednolicić linie mail vs `latest.json`** (kamień z plan.md): jedno ziarno
  `seed=int(YYYYMMDD)` w obu ścieżkach.
- [ ] **`--redo-draws` na 93 roll-downach** (parser naprawiony w PR #11, dane wciąż
  zepsute) — po §1 mniej pilne (sprzedaż przyjdzie z Merseyworld), ale nadal potrzebne
  do walidacji dokładnej reguły roll-downu (§3).
- [ ] **Cron-y GitHub Actions chodzą ~1–2,5 h późno** (zmierzone w plan.md): przesunąć
  crony o godzinę wcześniej względem pożądanego czasu i/lub dodać drugi strzał.
- [ ] Merseyworld/beatlottery jako **drugie źródło w watchdogu** (dziś podpowiada tylko
  scraper archiwum).

## 8. Czego świadomie nie robimy — z dowodami

| Pomysł | Dlaczego nie | Źródło |
|---|---|---|
| LSTM/transformer na liczbach | brak sygnału do nauczenia; jedyny rzetelny backtest przegrał z randomem | Haigh 1997; PMC5536828; Medium/J. David |
| „Gorące/zimne" liczby | χ² zgodne z uniformem na 2 231 losowaniach | Joe 1993; Genest 2002 |
| Kupowanie puli (buy-the-pot) | wymaga ~£90M (C(59,6)×£2), ryzyko podziału (Irlandia 1992: 3-way split) | Moffitt & Ziemba; RTE/Irish Times |
| Granie poza MBW | próg zwykłego losowania £30,2M vs p99 puli £28,7M | własne dane, plan.md |

## 9. Kolejność i szacunek

| Krok | Zależy od | Czas | Efekt |
|---|---|---|---|
| §1 sprzedaż Merseyworld | — | 1–2 wieczory | realny N per losowanie, 3 195 obserwacji |
| §2 api-dfe JSON | — | 1 wieczór | trwały feed + fundusze per tier |
| §4 model sprzedaży | §1 | 2–3 wieczory | uplift → predykcja z przedziałem |
| §3 dokładny roll-down | §7 redo-draws | 2 wieczory | precyzja na granicznych pulach |
| §5 popularność B&M | — | 1–2 tygodnie wieczorami | najlepszy publikowalny komponent projektu |
| §6 Kelly | §3 | 1 wieczór | „ile grać" w mailu |
| §7 higiena | — | 1 wieczór | mniej długu |

**Miara sukcesu pozostaje ta sama co w plan.md:** nie „więcej trafień", tylko werdykt
PLAY/SKIP, który przeżywa wrażliwość na N, oraz ledger ROI z realnymi kuponami.
Realistycznie: ~4 okazje +EV rocznie, przewaga rzędu dziesiątek pensów na linii —
world-best znaczy tu „najdokładniejsza wycena na świecie", bo lepszej obietnicy
matematyka nie dopuszcza.
