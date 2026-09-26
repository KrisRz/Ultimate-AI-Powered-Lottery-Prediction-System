# Architektura — co mamy, gdzie i jak działa

Mapa całego systemu. **Zacznij tu** w każdej nowej sesji, potem `CLAUDE.md`
(zasady), potem `audit-2026-09-05.md` (kanon analizy i historia decyzji).
Stan na **2026-09-26**, `main` = f1d7c02 (PR #44–#48). Kiedy coś tu przestanie
być prawdą — popraw ten plik w tym samym commicie.

---

## 0. W jednym akapicie

Toolkit EV dla UK Lotto, **nie predyktor**. Odpowiada na dwa pytania: **kiedy
grać** (czy to losowanie ma dodatnią wartość oczekiwaną?) i **czym grać**
(które linie dzielą jackpot z najmniejszą liczbą ludzi). Najczęstsza poprawna
odpowiedź to SKIP; PLAY zdarza się 1–2 razy w roku (Must-Be-Won w środę przy
puli ~£9M albo specjal ≥ ~£17M). Kul nie da się przewidzieć i projekt to
zmierzył (fairness p=0,600, backtest bez przewagi). Historia służy do
kalibracji tego, co ludzie grają, ile kupują i ile płacą nagrody.

---

## 1. Mapa systemu

```
                           ┌────────────────────────────────────────────┐
  AWS EventBridge ────────▶│ GitHub Actions (chmura, źródło prawdy)     │
  (punktualny zegar)       │                                            │
   śr/sob 21:50 collect    │  collect.yml  zbiera → gate → EV → mail    │──▶ commit data/ + site.json
   czw/nd 06:05 collect    │  watchdog.yml czy kolektor w ogóle ruszył  │──▶ mail przy awarii
   czw/nd 12:05 watchdog   │  ticket.yml   kupony na żądanie → mail     │──▶ mail z liniami
                           │  stats.yml    raporty statystyczne (1. dnia)│
  GitHub cron (spóźnia się │  ci.yml / site-ci.yml / infra-plan.yml     │
  2–11 h, zapasowy)        │  site-deploy.yml → S3 + CloudFront         │──▶ lotto.krisgrzepka.com
                           └────────────────────────────────────────────┘
                                          │ git pull
                                          ▼
                           ┌────────────────────────────────────────────┐
                           │ Mac (lokalnie)                             │
                           │  ./lotto            terminal + menu        │
                           │  make play/ticket/wheel/roi/dashboard      │
                           │  launchd czw/nd 09:00 → post_draw.sh       │
                           │  data/ledger.csv  (TYLKO tu) ──aws s3 cp──▶│ S3 lotto-ledger-backup-<acct>
                           └────────────────────────────────────────────┘
```

**Zasada podziału:** chmura zbiera, liczy werdykt i wysyła maile — Mac bywa
wyłączony. Mac trzyma rejestr prawdziwych kuponów (repo jest publiczne) i jest
miejscem pracy (`./lotto`, analizy).

---

## 2. Harmonogram (UTC)

| kiedy | co | skąd |
|---|---|---|
| śr/sob ~19:00 | losowanie (sprzedaż zamyka się 19:30 Londyn) | operator |
| śr/sob 21:45 | `collect.yml` (cron GitHuba, zwykle późno) | `.github/workflows/collect.yml` |
| śr/sob 21:50 | `collect.yml` (dispatch z AWS, punktualnie) | `infra/modules/draw-trigger` |
| czw/nd 06:00 / 06:05 | retry kolektora; **niedziela 06:05 = mail statusowy** | jw. |
| czw/nd 09:00 | `post_draw.sh` na Macu (sync, rozliczenie rejestru, dashboard) | `ops/com.lotto.postdraw.plist` |
| czw/nd 12:00 / 12:05 | `watchdog.yml` — czy losowanie zostało zebrane | jw. |
| 1. dnia miesiąca 04:00 | `stats.yml` — fairness, ensemble, raporty | `.github/workflows/stats.yml` |
| na żądanie | `ticket.yml` — linie mailem | przycisk Run workflow |

`concurrency: collect` kolejkuje dwa zegary; drugi run nie znajduje nic nowego
i nic nie commituje (ingest jest idempotentny).

---

## 3. Kolektor krok po kroku (`collect.yml`)

1. **Fetch** — `scripts/fetch_data.py::download_fresh_data()`: API JSON
   `api-dfe` (główne), XML jako fallback. Samonaprawa luk do ~180 dni wstecz.
   ⚠️ `results/1/{n}` to stara gra jednorundowa (≤3178) — dwurundowe są pod
   `results/6/{n}`.
2. **Pule** — `scripts/backfill_draw_pools.py` (idempotentny).
3. **Scorecard Must-Be-Won** — `scripts/monitoring/post_mbw_validation.py`
   → `data/mbw_validation.csv` (prognoza sprzedaży vs zmierzona).
4. **Snapshot strony** — `scripts/export_site_data.py` → `site/public/data/site.json`
   + `site/src/__fixtures__/popularity-golden.json` (deterministyczny).
5. **Commit + push** danych i snapshotu; jeśli strona się zmieniła →
   `gh workflow run site-deploy.yml` (push z GITHUB_TOKEN nie wyzwala eventów).
6. **Dead-man ping** — no-op, dopóki nie ma sekretu `HEALTHCHECK_URL`.
7. **Pre-EV gate** — `scripts/monitoring/pre_ev_gate.py`: blokuje tylko to, co
   psuje DZISIEJSZY werdykt (brak puli, NaN jackpot, sprzeczna tożsamość
   losowania, braki w tierach, stałe modelu).
8. **EV + mail** — `scripts/monitoring/ev_alert.py` (patrz §6).
9. Po mailu, nigdy przed nim: **data contract** (`scripts/data_contract.py`:
   KNOWN / UNKNOWN_BUT_VALID / INVALID), **świeżość**, **pule nadążają**,
   **cały pytest** na nowych danych.

Zasada kolejności: nic, co dotyczy tylko archiwum, nie może zablokować maila.

---

## 4. Dane (`data/`)

| plik | co | kto pisze | uwagi |
|---|---|---|---|
| `lotto_full_history.csv` | wszystkie losowania 1994→, obie rundy | kolektor | `Jackpot` to NIE pula (starsze = reklamowany szacunek) |
| `merged_lottery_data.csv` | era 59 kul, runda 1 | kolektor | świeżość |
| `prize_tiers.csv` | tiery nagród od 3190 + `next_jackpot_*`, `rollover_count` | kolektor | źródło warunków następnego losowania |
| `draw_pools.csv` | pula każdego losowania od 3179 | kolektor | **sprzedaż = (pula − poprzednia) / 8,88%** — tożsamość, nie szacunek |
| `mbw_validation.csv` | scorecard Must-Be-Won | kolektor | |
| `prize_tiers_history.csv` | backfill tierów 2066–3195 | zamrożony | |
| `sales_history.csv` | sprzedaż 1994–3195 (Merseyworld) | zamrożony | okrągłe £1M = placeholder |
| `ledger.csv` | **prawdziwe kupony** | Mac (`roi_ledger.py`) | gitignored, repo publiczne; kopia w S3 |

⛔ **Kolektor jest właścicielem `data/`.** Lokalne kopie plików kolektora nigdy
nie są warte zachowania — `sync_collector_data.sh` je odrzuca. Konfliktów nie
rozwiązuje się ręcznie.

**Rosnące archiwum** — `scripts/archive.py`: `load_tier_archive()` i
`load_sales_archive()` doklejają do zamrożonych backfilli wszystko, co kolektor
zebrał po 3195. Używają ich: `rolldown_history.py`, `calibrate_mbw_uplift.py`,
`backtest_wheel.py`. ⛔ **Nie** używa go `calibrate_popularity.load_joined` —
zamrożony eksperyment popularity-v2 wymaga, by kończył się na 3195.

---

## 5. Silnik obliczeń — `lottery/ev.py` (jedno źródło prawdy)

Każda stała ma w komentarzu pochodzenie — przeczytaj go, zanim zmienisz liczbę.

| technika | funkcje | sedno |
|---|---|---|
| **hipergeometria** | `P_MATCH_*` | jackpot 1:45 057 474 na rundę; 2 rundy/kupon |
| **sprzedaż** | `exact_lines_sold`, `exact_sales_baseline`, `estimate_tickets_sold` | tożsamość z puli; baza z tego samego dnia tygodnia; fallback: winner counts (±15%) |
| **uplift Must-Be-Won** | `MBW_UPLIFT_BY_WEEKDAY` (śr ×1,44, sob ×1,27), `SPECIAL_MBW_UPLIFT_BY_WEEKDAY` | ⛔ nie obniżać bez n ≥ 4 i punktu z danego dnia tygodnia |
| **popularność linii** | `number_weight`, `popularity_ratio` | 3 przedziały 1.23/1.10/0.83 + kary za wzory, znormalizowane do średniej 1,0 |
| **dzielenie jackpota** | `expected_cowinner_share` | E[1/(1+K)], K ~ Poisson; runda 1 = twoje liczby, runda 2 = liczby tamtego losowania |
| **roll-down** | `rolldown_tier_boosts` | £5 do Match 2 najpierw, reszta Match 3 (Game Procedures) |
| **nagrody stałe** | `calibrate_fixed_prizes` | mediana z ostatnich 30 losowań |
| **EV linii** | `line_ev`, `break_even_jackpot`, `should_play` | EV = Σ P×nagroda − £2 |
| **odporność** | `sales_sensitivity`, `decision_stability` | p25/p75 sprzedaży; model płaski vs podwojony → ROBUST PLAY/SKIP, MODEL-SENSITIVE |
| **druga opinia** | `exact_era_uplifts`, `measured_uplift`, `at_measured_uplift` | ta sama cena przy upliftcie ZMIERZONYM od czerwca 2026 |
| **klasyfikacja** | `classify(verdict, measured)` | **PLAY / MARGINAL / SKIP** — jedyna definicja w systemie |
| **kalendarz** | `upcoming_draw_date`, `last_closed_draw_date`, `forecast_must_be_won`, `must_be_won_outlook` | liczone wobec 19:30 Londyn, nie wobec daty |
| **literatura** | `abrams_garibaldi_screen`, `kelly_stake` | drugie zdanie; Kelly = grosze przy detalicznym kapitale |

`lottery/portfolio.py::build_portfolio` — losowanie ważone „w stronę
niepopularnych” + zachłanny wybór najwyższego EV; ograniczenia: nakładanie ≤2,
≥2 liczby >31, suma 100–260, bez trzech kolejnych.

`scripts/wheel_play.py::wheel_portfolio` — koło (12,6,4,3): 6 kuponów na 12
najrzadziej granych liczbach; gwarancje mierzone wyczerpująco.

---

## 6. Warstwa decyzji i jej wyjścia

```
next_draw_conditions()  (scripts/ev_play.py — czyta prize_tiers + draw_pools)
        │
        ▼
advise()  → Advice (frozen): cond, verdict, advice=PLAY|MARGINAL|SKIP,
        │                    measured, outlook, stale, notes, portfolio
        ├── render()  → wydruk `make play` / `./lotto play`
        ├── save()    → outputs/predictions/latest.json  (+ metadata.provenance:
        │               advice, draw_date, git_sha, git_dirty); what-if nie zapisuje
        ├── ev_alert.main()   → mail PLAY / MARGINAL / niedzielny status
        ├── ticket_mail.py    → mail z liniami na żądanie
        └── scripts/lotto.py  → terminal
```

`advise()` jest odporny: awaria drugiej opinii (MARGINAL, outlook) → notatka,
nigdy wyjątek — mail PLAY musi przejść przez wszystko.

**Maile** (SMTP z sekretów repo, `scripts/monitoring/notify.py`):

| mail | kiedy | kod |
|---|---|---|
| `LOTTO +EV ALERT: PLAY …` | werdykt PLAY | `ev_alert.build_alert` |
| `LOTTO MARGINAL: …` | SKIP w modelu, PLAY przy zmierzonym upliftcie (tylko cap-MBW) | `ev_alert.build_marginal_alert` |
| `LOTTO weekly: OK/DATA BEHIND …` | niedziela, tylko run z EventBridge (`workflow_dispatch`, <10:00 UTC) | `ev_alert.build_heartbeat` |
| `LOTTO ticket: 5 lines …` | przycisk w `ticket.yml` | `ticket_mail.build_ticket_mail` |
| awaria kolekcji | `watchdog.yml` | `collection_watchdog.py` |

**Brak niedzielnego maila = kolektor nie doszedł do kroku alertu.**

---

## 7. Terminal — `./lotto` (`scripts/lotto.py`, `make lotto`)

Cienka warstwa, nie liczy niczego sama.

| komenda | robi |
|---|---|
| `./lotto` | menu (status + opcje) |
| `status` | następne losowanie, świeżość, gate, werdykt, MBW |
| `play` | dokładnie `make play` (test porównuje z golden) |
| `ticket [--yes] [--record]` | 5 linii; przy SKIP pyta, ze skryptu wymaga `--yes` |
| `wheel [--yes] [--record]` | koło 6 kuponów |
| `check 3 7 12 19 24 31` | twoje liczby: popularność, % jackpota, EV |
| `whatif --jackpot … --roll-down` | hipotetyczne losowanie, nic nie zapisuje |
| `ledger [report\|settle\|add]` | rejestr |
| `analysis [nazwa]` | tylko raporty read-only (lista w `ANALYSES`) |
| `sync` | `sync_collector_data.sh` — tylko na czystym `main` |

Z telefonu: GitHub → Actions → **Lotto ticket (email me lines)** → Run workflow
(`portfolio`/`wheel`, `lines`, `variant`).

---

## 8. Rejestr kuponów

- `scripts/roi_ledger.py add|settle|report`; `data/ledger.csv` tylko na Macu.
- Każdy wiersz niesie **provenance z zapisanego werdyktu**: advice, git_sha
  (z `+dirty`), EV, próg, stabilność, pula, sprzedaż. Werdykt dla innego
  losowania jest odrzucany. `git_sha` czytany jako tekst.
- Rozliczanie: launchd czw/nd 09:00 (`post_draw.sh`) albo `make roi`.
- **Kopia:** S3 `lotto-ledger-backup-<account>` (eu-west-2, wersjonowany,
  AES256, public access zablokowany). Na razie **ręcznie** po zmianie:
  `aws s3 cp data/ledger.csv "s3://$(cd infra/live && terraform output -raw ledger_backup_bucket)/ledger.csv"`.
- Stan 2026-09-26: 15 linii, 2 losowania, £30 → £6 (−80%).

---

## 9. Analizy i walidacja (make)

| target | co | czas |
|---|---|---|
| `uplift` | uplift MBW: zainstalowany vs zmierzony na dokładnych pulach | s |
| `fairness` | czy maszyny są uczciwe (6 testów) | min |
| `ensemble` / `ensemble-null` | czy jakakolwiek metoda bije losowanie; null na syntetycznych historiach | min / dłużej |
| `popularity-audit` | czy model popularności trzyma się danych | ~1 min |
| `contract`, `pre-ev` | kontrakt danych, bramka | s |
| `popularity-v2-verify/power/selftest` | zamrożony eksperyment v2 (⛔ ewaluatora końcowego nie uruchamiać ponownie) | |
| `site-data` / `site-data-check` | snapshot strony | s |
| `test` | pytest (~515, ~30 s) | |

Bez targetu: `rolldown_history.py`, `backtest_wheel.py` (dostępne przez `./lotto analysis`).

---

## 10. Strona — lotto.krisgrzepka.com (`site/`)

Next.js, statyczny eksport, S3 (prywatny, OAC) + CloudFront, deploy z Actions
przez OIDC. Sekcje w `site/src/sections/`: S1Hook, S2Predict (zamrożony backtest
z `site/data-src/backtest.json`), S3Ev, SGenerator (panel A), SWhyNumbers,
SRolldown, SWheel, SMoney (panel F — rejestr z `site/data-src/ledger.json`,
odświeżany `export_site_data.py --refresh-ledger`), SBuilt.

Model popularności istnieje **dwa razy** — Python (`lottery/ev.py`) i TypeScript
(`site/src/data/`), bo linie generuje przeglądarka. Pilnuje ich
`popularity-golden.json`. Zmieniasz jedno → zmieniasz drugie.

Kontrole (wszystkie pięć, w tej kolejności, jak Site CI):
`cd site && npm run lint && npm run typecheck && npm test && npm run build && npm run size`.

---

## 11. Infrastruktura — `infra/` (Terraform, konto AWS Krzysztofa)

| ścieżka | co |
|---|---|
| `bootstrap/` | bucket stanu (stan lokalnie) |
| `live/` | strefa Route 53 (lookup), ACM us-east-1, moduł strony, DNS, rola deploy, draw-trigger, budżet, **bucket kopii rejestru** |
| `modules/static-site` | bucket, OAC, CloudFront |
| `modules/github-oidc` | provider + jedyna rola dla CI |
| `modules/draw-trigger` | EventBridge → API destination → GitHub dispatch; DLQ, alarmy → SNS → mail |

⛔ **Zawsze `-target=`**, nigdy pełny plan/apply (drift). ⛔ **Nigdy
`terraform test`** — mock_provider trafił kiedyś do prawdziwego konta. Po apply
weryfikuj poza Terraform (`aws s3api …`, `dig`).

---

## 12. Testy i zabezpieczenia

- **Golden** (`tests/golden/`, `tests/test_ev_play_golden.py`,
  `tests/test_wheel_play_golden.py`): wydruk bajt w bajt i liczby werdyktu na
  danych do 3209 i zamrożonym zegarze. Regeneracja
  (`UPDATE_GOLDEN=1`) tylko przy świadomej zmianie zachowania — opisanej w commicie.
- `tests/test_lotto_cli.py` — terminal = `make play` / `make ticket`.
- Test czytający zebrany plik **przypina snapshot** (`.query("draw_number <= N")`).
- `export_site_data.py --check` w CI: zmiana kodu, która rusza opublikowaną
  liczbę, nie wejdzie bez `make site-data` w tym samym commicie.
- `site.json` zapisuje liczbę testów → konflikty rozwiązuj `make site-data`, nie ręcznie.
- Testy ev_alert czyszczą `GITHUB_EVENT_NAME` (inaczej w niedzielę rano wszedłby heartbeat).

---

## 13. Sekrety i konfiguracja (tylko nazwy)

- Repo secrets: `SMTP_SERVER`, `SMTP_USER`, `SMTP_PASS`, `EMAIL_TO`
  (`HEALTHCHECK_URL` — brak, dead-man nieuzbrojony).
- Repo variables: `SITE_BUCKET`, `SITE_DISTRIBUTION_ID`, `SITE_DOMAIN`, `AWS_DEPLOY_ROLE_ARN`.
- Mac: `~/.lotto_env` (SMTP dla lokalnego alertu), AWS CLI (profil z prawem do S3).
- Interpreter: `./conda-py311/bin/python`, zawsze `PYTHONPATH=.` (Makefile i `./lotto` ustawiają).

---

## 14. Czego NIE robić

- Przewidywania kul w jakiejkolwiek postaci (LSTM usunięty w C1, −2727 linii).
- Obniżania `MBW_UPLIFT_BY_WEEKDAY` przed n ≥ 4 (dziś n=3: 1,127 / 1,023 / 1,152).
- Instalowania gładkiego modelu popularności (v2 = INCONCLUSIVE) ani ponownego uruchamiania jego ewaluatora.
- Ręcznego rozwiązywania konfliktów w plikach kolektora.
- Zapisu `latest.json` z what-if; commitowania `data/ledger.csv`.
- Gry poza Must-Be-Won / specjalami jako „inwestycji”.
- PR-ów piętrowych bez odczekania na przestawienie bazy (#46 wylądował w `chore/cleanup`).

---

## 15. Otwarte sprawy (2026-09-26)

1. Pierwszy **niedzielny mail statusowy z chmury** — sprawdzić run ~2026-09-27 06:05 UTC.
2. **Automatyczna kopia rejestru do S3** po `add`/`settle` — czeka na uprawnienie `Bash(aws s3 cp:*)`.
3. **`HEALTHCHECK_URL`** — dead-man ping wymaga konta healthchecks.io.
4. **Artefakt środa/sobota w kalibracji popularności** (68% wariancji reszt) — osobny, falsyfikowalny PR.
5. **Rozkład dzielenia jackpota** w werdykcie (P(dzielony), kwantyle) — `line_return_distribution` już jest.
6. Najbliższe capped MBW ~sob 2026-10-17 → SKIP (−£0,43 / zmierzony −£0,35).

---

## 16. Dokumenty

| plik | rola |
|---|---|
| `ARCHITECTURE.md` | ta mapa — co i gdzie |
| `CLAUDE.md` | zasady pracy w repo |
| `README.md` | użycie (komendy, maile, terminal) |
| `audit-2026-09-05.md` | **kanon**: analiza, backlog, historia decyzji (§9–§13) |
| `plan.md`, `plan-ulepszen-2026-08.md` | poprzednicy audytu |
| `popularity-v2-spec.md` | zamrożona specyfikacja eksperymentu |
| `OPTIMIZATION_PLAN.md` | nieaktualny, zostawiony dla linków |
