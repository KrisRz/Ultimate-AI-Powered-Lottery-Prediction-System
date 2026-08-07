# Plan ulepszeń — „world-best" toolkit EV dla UK Lotto (2026-08-07)

Research z 2026-08-07: trzy niezależne kwerendy (literatura naukowa o +EV w loteriach,
źródła danych UK Lotto zweryfikowane żywymi fetchami, przegląd ML-owych repo na GitHubie).
Wszystkie URL-e w tym pliku były sprawdzone tego dnia. Uzupełnia `plan.md` (FAZA 7);
niczego z niego nie unieważnia.

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
- [ ] Dodać ingest JSON jako główny (XML zostaje fallbackiem), zapisywać
  `prizeFundCents`/`prizeCap` per tier — od razu lepsza walidacja modelu roll-downu.
- [ ] Watchdog: sprawdzać oba źródła (api-dfe + Merseyworld archive) zanim uzna
  losowanie za stracone.

## 3. PRIORYTET 1 — Dokładna reguła roll-downu zamiast J/N 🎯

Z reguł gry (potwierdzone przez lottery.co.uk/lotto/must-be-won-draws i komunikat TNL):
roll-down **najpierw płaci £5 każdemu zwycięzcy Match 2, a dopiero resztę rozdziela
między zwycięzców Match 3** — nie proporcjonalnie po wszystkich tierach i nie jednym
strumieniem J/N. Nasze przybliżenie zwalidowało się na 3190 w 2% (J/N = £1,28 vs £1,26),
ale dokładna reguła daje:

- [ ] `rolldown_ev` rozbite na tiery: stały bonus `+£4 × P(match2) × rounds` plus
  człon Match 3 `(J − 4·E[W₂]) / E[W₃]` × P(match3). Ważne przy małych pulach (jak
  £8,39M z 08.08), gdzie proporcje tierów decydują o werdykcie na marginesie.
- [ ] Walidacja na przeparsowanych roll-downach z archiwum (patrz §5) — czy podział
  M2-najpierw-M3-reszta odtwarza obserwowane £24/£5, £24 = 10 + 14 itd.
- [ ] Specjalne losowania MBW (Allwyn planuje ~£15M na święta **bez** 5 rolloverów;
  roll-down identyczny): `forecast_must_be_won` liczy tylko cap-driven — flaga
  `next_jackpot_roll_down` je złapie, ale dashboard powinien odróżniać te dwa typy.

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

**Zadania:**
- [ ] Refit wag z likelihoodem NB + łączna inferencja z M2/M3/M4 (dane są).
- [ ] Prototyp modelu klastrowego B&M 2011; porównanie out-of-sample z obecnym
  (metryka: log-likelihood winner-counts na hold-oucie, nie „ładność”).
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

- [ ] Rozkład EV per tier już w modelu jest — policzyć z niego frakcję Kelly'ego
  (i ½-Kelly) dla zadanego majątku/budżetu; do maila PLAY dołączać **sugerowaną liczbę
  linii** zamiast stałych 5. Uczciwie: przy naszej skali (£10–20/losowanie) to głównie
  walor edukacyjny — ale to jest różnica między „alertem" a „systemem decyzyjnym".
- [ ] Szybki test analityczny Abrams & Garibaldi (`s(p,N) = [1−(1−p)^N]/N`, reguła
  `N < J/5`) jako niezależny sanity-check werdyktu —
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
