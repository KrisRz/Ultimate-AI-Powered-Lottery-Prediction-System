# popularity-model-v2 — zamrożona specyfikacja challengera

**Status: ZAMROŻONE 2026-09-19, na `main` w `17746e0`, PRZED odczytaniem
jakiegokolwiek wyniku na wycinku testowym.**

Ten dokument jest preregistracją, nie notatką. Powstał po to, żeby po
zobaczeniu wyniku finalnego testu nie dało się — także nieświadomie — zmienić
definicji modelu, wycinka, metryki ani reguły decyzyjnej. Kod trzymający te
same liczby: `scripts/validations/popularity_v2_frozen.py`. Pilnuje ich
`tests/test_popularity_v2_frozen.py` (SHA-256 pliku + refit z archiwum).

Kontekst: `audit-2026-09-05.md` §12.3. Trzy kubełki przegrywają
out-of-sample z modelem gładkim (0,11432 vs 0,10293), stabilnie na splitach
3–10, a pełna krzywa gładka zmienia zwykłe losowanie £32M ze SKIP na PLAY.
**Nic z tego nie jest jeszcze dowodem, że gładki należy zainstalować** — to
jest materiał, na którym gładki został *wybrany*.

---

## 1. Challenger — pełna definicja

**Postać.** Regresja liniowa z dwoma zawiasami (hinge), bez stopni:

```
log(mult_i) = b0 + b1 · Σ n + b2 · Σ max(n − 12, 0) + b3 · Σ max(n − 31, 0)
```

gdzie suma biegnie po sześciu wylosowanych liczbach obserwacji `i`
(obserwacja = para (losowanie, runda)).

**Węzły: 12 i 31. Nie były szukane.** To są granice kalendarzowe, których
model produkcyjny używa od 2026-07-25 (`lottery/ev.py: number_weight` —
miesiące 1–12, dni 1–31). Żaden inny zestaw węzłów nigdy nie był dopasowany
ani porównany. **Challenger nie ma ani jednego strojonego hiperparametru** —
to jest istotne przy czytaniu wyniku: nie ma stopnia swobody, na który dałoby
się zrzucić wygraną albo przegraną.

**Zmienna objaśniana.** `mult` z `scripts/calibrate_popularity.py`:
`raw = winners / P_MATCH_3`, trend sprzedaży = krocząca mediana `raw`
w oknie **51** obserwacji (centrowana, `min_periods=25`),
`mult = raw / trend`. Tier **5 = Match 3** (~130 tys. zwycięzców — najlepszy
stosunek sygnału do szumu). Obserwacje z `winners > 0`.

**Estymator.** `numpy.linalg.lstsq(X, y, rcond=None)` — zwykły OLS, bez wag,
bez regularyzacji, bez odrzucania obserwacji odstających.

**Z współczynników na wagi per liczba.**

```
log w(n) = (6/3) · (b1·n + b2·max(n−12,0) + b3·max(n−31,0))
w(n)     = exp(log w(n) − mean(log w))          # centrowanie w logach
w(n)     = w(n) · 59 / Σ w                       # średnia populacyjna = 1,0
```

Rozjaśnienie `6/3 = 2` jest tłumaczeniem kalibracji: zwycięzca Match 3 dzieli
3 z 6 wylosowanych liczb, więc log-mnożnik niesie połowę sumy log-wag.
Normalizacja do średniej 1,0 to ograniczenie populacyjne — sumaryczna
popularność musi się zgadzać z liczbą kuponów.

**Dane treningowe.** Wszystkie 1147 obserwacji, losowania **2066–3195**
(2015-10-10 → 2026-08-05), z `data/lotto_full_history.csv` +
`data/prize_tiers_history.csv`. To jest cały materiał, którego challenger
użył do tej pory — i cały materiał, jaki dostanie.

**Zamrożone współczynniki** (dopasowanie na 2066–3195):

| | b0 | b1 (Σn) | b2 (hinge 12) | b3 (hinge 31) |
|---|---|---|---|---|
| smooth | +0,56398574 | +0,00114579 | −0,00569932 | −0,00263229 |

| | b0 | #{n≤12} | #{n>31} |
|---|---|---|---|
| 3-bucket | +0,33581754 | +0,05605431 | −0,13848899 |

**Zamrożona krzywa** (59 wartości w `SMOOTH_WEIGHTS`): 1,187 dla kuli 1,
szczyt **1,2173 na kuli 12**, potem monotoniczny spadek do **0,6847** na
kuli 59. Rzut na trzy kubełki: 1,202 / 1,113 / 0,837.

---

## 2. Incumbent — z czym się mierzy

Ten sam estymator, te same wiersze, projekt:
`log(mult) = b0 + b1·#{n≤12} + b2·#{n>31}` (13–31 jest poziomem odniesienia).

Refit daje **1,2302 / 1,0997 / 0,8337** wobec **1,23 / 1,10 / 0,83**
zainstalowanych w `lottery/ev.py`. Różnica jest w czwartej cyfrze, więc test
mierzy się z modelem produkcyjnym, nie z jego przybliżeniem.

---

## 3. Co jest naprawdę nietknięte — i dlaczego

Uczciwa odpowiedź ma trzy poziomy, nie jeden.

### 3.1 PROSPECTIVE HOLDOUT: losowania 3196–3207 — jedyny naprawdę nietknięty

`load_joined()` czyta `data/prize_tiers_history.csv`, który **kończy się na
3195**. Rozbicie tierów dla losowań 3196–3207 istnieje wyłącznie w pliku
kolektora `data/prize_tiers.csv`. Te wiersze nigdy nie weszły do żadnego
dopasowania, żadnego splitu i żadnego rankingu specyfikacji — nie brały
udziału w wyborze postaci gładkiej, węzłów, tiera, estymatora, okna trendu
ani liczby splitów, bo w momencie tamtych decyzji fizycznie nie były
widoczne dla kodu, który je podejmował.

**PRIMARY (16 wierszy):** 8 losowań, którym tożsamość puli potrafi policzyć
dokładną sprzedaż (`lottery.ev.exact_lines_sold`), × 2 rundy:

```
3196, 3198, 3199, 3201, 3202, 3203, 3204, 3205
```

Brakujące 3197, 3200, 3206, 3207 to losowania po wygranej albo po puli
ustawionej przez operatora — tożsamość ich nie wycenia (CLAUDE.md: „The pool
identity needs a rollover, not just a bigger pool").

Mnożnik liczony **dokładnie**, bez kroczącej mediany:
`mult = winners / (P_MATCH_3 · N)`, gdzie `N` = dokładna liczba linii z puli.
To jest lepszy mianownik niż ten, na którym oba modele były trenowane —
i identyczny dla obu modeli, więc porównanie pozostaje symetryczne.

**SECONDARY (24 wiersze):** wszystkie 12 losowań 3196–3207 × 2 rundy, mnożnik
z kroczącej mediany jak w treningu (na ogonie jednostronnej, bo brakuje
przyszłych obserwacji). Raportowany **zawsze**, nigdy rozstrzygający —
zapisane tutaj po to, żeby nie dało się wybrać korzystniejszego wariantu po
fakcie.

⚠️ **Dwa z ośmiu losowań primary nie są w pełni dziewicze — i muszę to
powiedzieć.** `data/mbw_validation.csv` ocenia losowania Must-Be-Won,
porównując dokładną sprzedaż z tożsamości puli z estymatorem z liczby
zwycięzców — a **iloraz tych dwóch liczb to dokładnie mnożnik popularności**,
który ten test punktuje. Dla **3196** audyt poszedł dalej i zapisał liczbę
(§3.2: winner-counts czytały 10,92M linii wobec 9,46M dokładnych, czyli
mnożnik ok. 1,15, „runda 2 była urodzinowa"). **3205** jest w scorecardzie
z tego samego powodu. Nikt nie porównywał na tych wierszach **specyfikacji** —
a to jest pytanie testu — ale człowiek widział ich poziom.

Ujawnione, nie wycięte: `HOLDOUT_SEEN_BY_SCORECARD = (3196, 3205)`, test
raportuje wynik z tymi wierszami i bez nich (16 vs 12 wierszy), a
preregistrowane weto czyta się z pełnych 16. Wycięcie ich po cichu byłoby
gorsze niż skażenie — i zostawiłoby 10% mocy zamiast 12%.

⛔ **Losowania od 3208 są świadomie wyłączone.** W chwili zamrażania 3208
jeszcze się nie odbyło. Zostaje jako wycinek replikacyjny na później —
wydanie go teraz nie zostawiłoby niczego, na co nikt nie patrzył.

### 3.2 RETROSPECTIVE VALIDATION: losowania 2066–2638 — nie holdout

`specification_contest` testował wyłącznie na wierszach `n//2 … n`
(`edges = linspace(n//2, n, splits+1)`), przy każdej liczbie splitów 3–10.
To znaczy: punktowany był materiał **2639–3195** (2021-04-07 → 2026-08-05,
574 wiersze). Pierwsza połowa — **2066–2638** (2015-10-10 → 2021-04-03, 573
wiersze) — była w każdym splicie wyłącznie materiałem treningowym.

⚠️ **To nie jest wycinek dziewiczy, nie wolno go tak nazywać i nie jest
samodzielną podstawą do instalacji modelu.** Te wiersze przesuwały
współczynniki w każdym dopasowaniu i weszły do kalibracji produkcyjnej
z 2026-07-25. Czego **nie** robiły: nigdy nie rankowały specyfikacji — ani
razu nie odpowiedziały na pytanie „który kształt jest lepszy".

Rola, jaką dostają: **mocna walidacja retrospektywna / reverse-temporal**
(dopasowanie na 2639–3195, punktacja wstecz na 2066–2638). Może obalić
challengera i może go wesprzeć, ale **sama nigdy go nie instaluje** — decyzja
Krzysztofa z 2026-09-19, i jest metodologicznie mocniejsza niż to, co
proponowałem.

Dla tego testu zamrożone są osobne współczynniki, dopasowane **tylko na
2639–3195** (`SMOOTH_BETA_SECOND_HALF`, `BUCKET_BETA_SECOND_HALF`).
Punktowanie pierwszej połowy współczynnikami z pełnego archiwum byłoby
wyciekiem, bo te współczynniki ją zawierają.

### 3.3 Poza zakresem: era 49 kul (losowania < 2066)

Inny zestaw kul — wagi dla liczb 50–59 nie istnieją, popularność „31" znaczy
co innego przy 49 kulach. Nie jest wycinkiem testowym i nigdy nie będzie.

---

## 4. Artefakt dnia tygodnia — znaleziony przy projektowaniu testu

Przy sprawdzaniu struktury zależności (wymóg: nie zakładać IID) wyszło coś,
czego nikt nie szukał. Autokorelacja reszt mnożnika kalibracji wynosi ±0,65
na **każdym** opóźnieniu, naprzemiennie. To nie jest zależność — to piła:

| dzień | obserwacji | średnia reszta |
|---|---|---|
| środa | 574 | **−0,257** |
| sobota | 573 | **+0,257** |

**68% wariancji reszt mnożnika to czysta różnica środa/sobota.**
`add_multiplier` dzieli przez kroczącą medianę w oknie 51 obserwacji, które
zawiera **oba dni tygodnia** — a mediana mieszanki nie usuwa różnicy poziomów
między nimi. Sobota sprzedaje ~1,67× tego co środa (e^0,514), więc każda
sobota siedzi nad medianą, a każda środa pod nią.

Co z tego wynika:

1. **Nie obciąża to dopasowania** — wylosowane liczby są niezależne od dnia
   tygodnia, więc β jest nieobciążone. Jest za to **nieefektywne**, a błędy
   standardowe kalibracji są zawyżone.
2. **Zmienia wielkość szumu, z którym mierzy się test.** Nie 0,11432, tylko
   **0,0320** po usunięciu dnia tygodnia. Moja wcześniejsza tabela mocy
   (12%) używała wariancji, która w dwóch trzecich jest artefaktem.
3. **Wycinek prospective jest wolny od artefaktu z definicji** — dzieli przez
   własną dokładną sprzedaż danego losowania, a nie przez medianę mieszanki.

Poprawka wpisana do evaluatora: wycinek retrospektywny ma **odjęte średnie
dniowe policzone na połowie fitującej (2639–3195)**, nigdy na punktowanych
wierszach. Jest wspólna dla obu modeli, więc nie może zmienić tego, który
wygrywa — może tylko zmniejszyć wariancję.

⚠️ To jest też realna, choć drobna wada `scripts/calibrate_popularity.py`
w produkcji. Do zapisania w audycie; **nie naprawiam jej w tej sesji**, bo
zmiana estymatora po zamrożeniu specyfikacji jest dokładnie tym, czego freeze
zabrania.

---

## 5. Co który wycinek jest w stanie rozstrzygnąć

Podłożona prawda = „krzywa gładka jest prawdziwa", szum **0,0320** (po
usunięciu dnia tygodnia), przedział pierwotny, jednostka = losowanie.
Systematyczne rozejście predyktorów **E[(μ_s − μ_b)²] = 0,01012**.

| wycinek | losowań | wierszy | mówi „supports smooth" |
|---|---|---|---|
| **prospective dziś** (3196–3205) | 8 | 16 | **19%** |
| prospective z trendem (3196–3207) | 12 | 24 | 31% |
| + 3 miesiące zbierania | 25 | 50 | 59% |
| + 6 miesięcy zbierania | 50 | 100 | **86%** |
| + rok | 100 | 200 | 99% |

(`make popularity-v2-selftest`, ziarno ustalone wewnątrz, błąd Monte Carlo
przy 600 powtórzeniach ≈ ±2 pkt proc.)

**Wycinek prospective ma dziś 19% mocy.** Dlatego brak istotności na nim
**musi** znaczyć INCONCLUSIVE, nie „wygrywa 3-bucket" — przy 18% mocy
milczenie jest wynikiem domyślnym, nie dowodem. Żeby prospective mógł
cokolwiek zainstalować sam, potrzebuje **ok. 50 losowań, czyli pół roku**
zbierania (2 losowania tygodniowo). To jest konkretny cel strumienia
replikacyjnego, nie ogólnik „zbierajmy dalej".

---

## 6. Evaluator i metryka — zamrożone przed uruchomieniem

`scripts/validations/popularity_v2_final_test.py`, przypięty SHA-256
w `tests/test_popularity_v2_final_test.py`. **Nic nie dopasowuje** — bierze
zamrożone wektory i stosuje je do wierszy, których nie widziały.

**Metryka pierwotna:**

```
d_i = (y_i − ŷ_bucket,i)² − (y_i − ŷ_smooth,i)²        d > 0 sprzyja smooth
```

Raportowane zawsze: **effect size** (średnie `d`, MSE obu modeli osobno),
**przedział**, liczba losowań sprzyjających każdemu modelowi, wersja
nieścentrowana i przesunięcie poziomu. p-value nie jest raportowane jako
kryterium — przedział jest.

**Przedział — wybrany na podstawie zmierzonego pokrycia, nie z gustu.**
Jednostką losowania (nie wiersz): rundy 1 i 2 tego samego losowania dzielą
mianownik sprzedaży, a ich reszty korelują **+0,91** na materiale treningowym.
Pokrycie przy prawdziwej hipotezie zerowej (równe MSE), dane syntetyczne
z wspólnym szokiem per losowanie:

| grup | bootstrap po wierszach | bootstrap po losowaniach | **t na średnich losowań** |
|---|---|---|---|
| 40 | 88% | 84% | 87% |
| 8 | 85% | 81% | **94%** |
| 300 (1 runda) | — | 89% (bloki 25) | 92% |

Bootstrap percentylowy **niedoszacowuje przy ośmiu grupach** — czyli dokładnie
tam, gdzie leży prospective. Dlatego:

- **prospective → przedział t na średnich per losowanie** (pierwotny),
- **retrospective → ruchomy bootstrap blokowy, blok 25 losowań** (pierwotny;
  573 losowania to dość grup, a krocząca mediana dzieli okno z sąsiadami),
- **oba przedziały drukowane zawsze**, plus bloki długości 1 / 25 / 51 dla
  retrospective. Długości bloku nie da się wybrać po zobaczeniu wyniku.

Zmierzona autokorelacja `d` **między** losowaniami: |ACF| < 0,11 na wszystkich
opóźnieniach 1–60. Zależność, która jest ogromna w resztach, w różnicy
parowanej **znosi się** — dlatego prostsza metoda jest tu uprawniona i dlatego
jest udokumentowana pomiarem, a nie założeniem.

**Dwie poprawki nuisance, obie wspólne dla modeli:** dzień tygodnia (§4,
tylko retrospective) i poziom (różnica poziomów między mnożnikiem
z dokładnej sprzedaży a mnożnikiem z mediany, ok. −0,05 w logach; usuwana
przez **punkt środkowy obu predykcji**, co jest symetryczne z konstrukcji).
Test syntetyczny pilnuje, że przesunięcie poziomu o 0,25 nie rusza `d`
ani o 1e-12.

---

## 7. Reguła interpretacji — zamrożona, trzy wyniki

Dozwolone są **trzy** odczyty. Żaden nie zmusza nas dziś do instalacji ani
do odrzucenia:

| warunek | odczyt |
|---|---|
| `d > 0` i przedział **nie zawiera** zera | **evidence supports smooth** |
| `d < 0` i przedział **nie zawiera** zera | **evidence supports incumbent** |
| przedział **zawiera** zero | **inconclusive — collect prospective evidence** |

⛔ **Przedział zawierający zero NIE jest zwycięstwem 3-bucket.** 3-bucket
zostaje zainstalowany, bo jest zainstalowany — to jest inne zdanie niż „dane
go wolą". Test `test_interpretation_rule_allows_three_outcomes` pilnuje tego
rozróżnienia w kodzie.

**Jak czytać trzy wycinki razem:**

| prospective (8 losowań) | retrospective (573) | wniosek |
|---|---|---|
| supports smooth | supports smooth | **najmocniejszy możliwy dziś wynik** → kalibracja i ogony, potem decyzja o instalacji |
| inconclusive | supports smooth | inconclusive; challenger żywy, 3-bucket dalej w produkcji, zbieramy 3208+ |
| inconclusive | inconclusive | inconclusive; challenger bez wsparcia, temat na później |
| supports incumbent | cokolwiek | **evidence supports incumbent** → temat zamknięty |
| supports smooth | supports incumbent | konflikt → inconclusive, zbieramy dalej |

Wycinek trend (3196–3207, 24 wiersze) jest raportowany zawsze i nie wchodzi
do tej tabeli.

⛔ Po odczytaniu wyniku: **żadnego refitu, żadnej zmiany węzłów,
współczynników ani normalizacji.** Historyczne przerzuty PLAY/SKIP dopiero
po kalibracji i ogonach — i tylko jeśli którykolwiek wycinek wesprze smooth.

⛔ `MODEL-SENSITIVE` / `ROBUST PLAY` / `ROBUST SKIP` zostają niezależnie od
wyniku.

⛔ **Losowania od 3208 to strumień replikacyjny.** Nie wchodzą do żadnego
projektowania modelu; evaluator odfiltrowuje je twardo
(`HOLDOUT_EXCLUDED_FROM`), a test pilnuje tego także po dzisiejszej zbiórce.

⚠️ **Zanim ten strumień zostanie kiedykolwiek odczytany, trzeba zamrozić
KIEDY wolno go odczytać.** Tabela mocy w §5 podaje ~25 losowań ≈ 59% i ~50 ≈
86% — to są orientacje, **nie pozwolenie na zaglądanie co kilka miesięcy**.
Zaglądanie wielokrotnie i instalowanie smooth przy pierwszym przedziale, który
przestanie obejmować zero, podnosi odsetek fałszywych pozytywów dokładnie tak
samo jak wybór przedziału po zobaczeniu wyniku — tylko rozłożone w czasie.
Replikacja wymaga albo **jednego z góry ustalonego terminu odczytu**, albo
z góry określonej metody sekwencyjnej (alpha spending / próg Pococka),
zapisanej razem z nową zamrożoną specyfikacją i powodem, zanim padnie
pierwsza liczba.

---

## 8. Jak jest pilnowane zamrożenie

- `make popularity-v2-verify` — odtwarza wszystkie cztery wektory
  współczynników z archiwum (dryf 0,000e+00 przy zamrażaniu).
- `make popularity-v2-power` — co który wycinek potrafi rozstrzygnąć.
- `make popularity-v2-selftest` — evaluator na danych syntetycznych:
  wykrywa podłożonego smooth (100%), podłożonego incumbenta (100%), milczy
  przy równym MSE (90%), pokrywa przy zależności.
- `tests/test_popularity_v2_frozen.py` — SHA-256 specyfikacji, współczynniki,
  normalizacja krzywej, definicja wycinka, zgodność dokumentu z kodem.
- `tests/test_popularity_v2_final_test.py` — SHA-256 evaluatora, trzy wyniki
  reguły interpretacji, wykrywanie obu podłożonych prawd, kalibracja pod
  hipotezą zerową, pokrycie przy zależności, niezmienniczość na poziom,
  uczciwość małej próby, twarde odfiltrowanie 3208+.
- Evaluator **odmawia drugiego uruchomienia**, jeśli plik wyniku już istnieje.

Jeśli po odczytaniu wyniku ktokolwiek zechce zmienić węzeł, postać,
normalizację, wycinek albo regułę — musi najpierw zmienić hash w teście.
O to chodzi.
