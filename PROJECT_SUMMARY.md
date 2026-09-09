# Podsumowanie projektu DEA-F

Stan dokumentu: 12 sierpnia 2026 r.

## Cel projektu

DEA-F generuje i ocenia odporne ścieżki poprawy (robust improvement paths) w Data Envelopment Analysis. Dla wybranej jednostki decyzyjnej (DMU) projekt szuka kolejnych, osiągalnych stanów o mniejszych nakładach i/lub większych efektach. Obliczenia łączą:

- Python — przygotowanie kandydatów, budowanie ścieżek, metryki, raporty i wizualizacje;
- Java/Maven — dokładne obliczenia robust DEA z biblioteki `dea-robustness`;
- CSV — dane wejściowe i wyniki kolejnych etapów eksperymentu.

Projekt implementuje pięć pipeline'ów: ścieżkę po diagramie Hassego, historyczną ścieżkę przez warstwy rankingu necessary, ścieżkę najlepszej rangi, ścieżkę najlepszej efektywności oraz ścieżkę zmniejszającą szerokość robustności. Iteracyjne fronty efektywności są obecnie wyznaczane w snapshotcie wejściowym i prezentowane w raporcie; nie należy ich utożsamiać z pipeline'em 2.

## Najważniejsze elementy repozytorium

| Ścieżka | Rola |
|---|---|
| `python/1_hasse_path_pipeline.py` | Metoda 1: przejścia po relacji necessary i diagramie Hassego. |
| `python/2_front_path_pipeline.py` | Metoda 2: historycznie nazwana ścieżka po warstwach rankingu necessary; nie jest peelingiem frontów efektywności. |
| `python/3_best_rank_path_pipeline.py` | Metoda 3: poprawa najlepszej osiągalnej rangi. |
| `python/4_best_efficiency_path_pipeline.py` | Metoda 4: poprawa najlepszej efektywności. |
| `python/5_robustness_width_path_pipeline.py` | Metoda 5: redukcja szerokości score lub rank. |
| `python/path_pipeline_common.py` | Wspólne generowanie kandydatów, osiągalność, selekcja przejść i zapis ścieżek. |
| `python/candidate_refinement.py` | Lokalna i globalna rafinacja kandydatów fictive. |
| `python/path_metrics.py` | Obliczanie metryk jakości każdej ścieżki. |
| `python/path_selection.py` | Pojedyncze, leksykograficzne, Pareto i ważone wybory ścieżek do raportu. |
| `python/validate_controlled_experiment.py` | Automatyczna kontrola spójności wspólnego eksperymentu pięciu metod. |
| `python/generate_five_method_robot_report.py` | Samowystarczalny raport HTML porównujący ścieżki dla Robot_18. |
| `python/collect_path_metrics.py` | Łączenie metryk wszystkich metod w podsumowanie eksperymentu. |
| `python/generate_dea_input_snapshot.py` | Bazowy snapshot `X/Y`, scores, rang, iteracyjnych frontów efektywności, macierzy NEC/POS, rankingu necessary i diagramu Hassego. |
| `python/render_path_metrics_*.py` | Raporty Markdown i PDF. |
| `python/visualize_*.py` | Wizualizacje 2D/3D i certyfikaty minimalności. |
| `java/` | Projekt Maven i klasy eksportujące metryki robust DEA do CSV. |
| `input/` | Dane wejściowe (`EDU.csv`, `airports.csv`) i dane pomocnicze. |
| `templates/` | Szablony danych, opisów metryk i przykładowego raportu. |
| `docs/EXPERIMENT_OUTPUT_AND_VISUALIZATION.md` | Kanoniczna instrukcja porównywalnego outputu 2D, certyfikatu, wykresu i tabeli. |
| `examples/five_unit_2d/` | Lekka, referencyjna wersja interaktywnego porównania pięciu DMU. |
| `input/robot_technology_table1.csv` | Dane 27 robotów z Table 1: koszt i powtarzalność jako inputy, udźwig i prędkość jako outputy. |
| `run_edu_3d_experiment.ps1` | Uruchomienie wszystkich pięciu metod dla eksperymentu EDU 3D. |
| `run_robot_2d_five_method_experiment.ps1` | Kontrolowany przykład pięciu metod dla Robot_18, dwóch inputów i stałych outputów. |
| `output/` | Lokalne wyniki eksperymentów; katalog jest ignorowany przez Git. |

Szczegółowy opis parametrów każdej metody i interpretacji wyników znajduje się w `README.md`.
Standard końcowego pliku z interaktywnym wykresem i krótką tabelą opisuje
`docs/EXPERIMENT_OUTPUT_AND_VISUALIZATION.md`.

## Aktualny kierunek zmian

Obecna wersja rozwija wybór kandydatów krok po kroku. Każdy kolejny punkt jest oceniany względem faktycznego poprzednika w danej częściowej ścieżce, a nie wyłącznie względem DMU startowego. Kandydaci muszą być osiągalni, po czym są redukowani do frontu Pareto minimalnych zmian. `--points-per-stage` ogranicza obecnie liczbę kandydatów osobno dla każdego przejścia.

Do projektu dodano także:

- `transition_candidates.csv` z kandydatami i wysiłkiem liczonym od poprzedniego punktu;
- automatyczny `path_metrics.csv` dla każdej metody;
- automatyczne `path_selection.csv`, `path_pareto.csv` i `path_pareto_representatives.csv` z audytowalnym wyborem ścieżki;
- wspólne zestawienia `all_path_metrics.csv` i `method_summary.csv`;
- rafinację lokalną oraz próbkowanie globalne dla metod numerycznych;
- raporty Markdown/PDF, wizualizacje 2D/3D i certyfikaty minimalności;
- wspólny snapshot wejścia DEA prezentowany przed ścieżkami: dane `X/Y`, scores, rangi, iteracyjne fronty efektywności, NEC/POS, ranking necessary i krawędzie Hassego;
- testy selekcji przejść, rafinacji i certyfikatu minimalności.

Koszt przejścia jest liczony wyłącznie po czynnikach faktycznie modyfikowanych przez eksperyment, ze stałą normalizacją względem pełnego zbioru referencyjnego. Pozostałe inputy i outputy nadal uczestniczą w kontroli osiągalności. Pipeline'y wspierają wagi czynników, etapów i stanów width oraz budżet łącznego wysiłku etapu i limity zmian poszczególnych czynników.

Każdy stan startowy i realny kandydat ma komplet skrajnych efektywności, rang i obu szerokości. `Cdir` jest obecnie przybliżeniem względem skończonej puli celów, z jawnym `cdir_scope`; nie jest analitycznym minimum w przestrzeni ciągłej. `PC` pozostaje puste bez rzeczywistych peer sets, natomiast pomocnicza ciągłość relacji necessary/possible ma osobną nazwę `RC`.

## Uruchomienie na nowym komputerze

### 1. Wymagania

- Git;
- Python 3.11 lub nowszy;
- JDK 21 lub nowszy (projekt Maven kompiluje kod dla Java 21);
- Apache Maven dostępny jako `mvn`;
- PowerShell 5.1 lub nowszy do skryptu zbiorczego.

### 2. Pobranie repozytorium i środowisko Pythona

```powershell
git clone https://github.com/Wo0kie3/DEA-F.git
Set-Location .\DEA-F
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r .\requirements.txt
```

Na Linux/macOS aktywacja środowiska to `source .venv/bin/activate`.

### 3. Instalacja biblioteki robust DEA

`java/pom.xml` używa artefaktu `io.github.alabijak:dea-robustness:0.1-SNAPSHOT`. Ponieważ jest to wersja `SNAPSHOT`, przed pierwszym uruchomieniem trzeba zainstalować bibliotekę w lokalnym repozytorium Maven:

```powershell
git clone https://github.com/alabijak/dea-robustness.git ..\dea-robustness
Set-Location ..\dea-robustness
mvn clean install
Set-Location ..\DEA-F
```

Kontrola konfiguracji:

```powershell
python --version
java -version
mvn --version
mvn -f .\java\pom.xml test
```

### 4. Szybki start

Najprostszy test pojedynczej metody na istniejących DMU:

```powershell
python .\python\1_hasse_path_pipeline.py `
  --input .\input\airports.csv `
  --target KAT `
  --output-dir .\output `
  --java-entry .\java `
  --mode real `
  --max-paths 20
```

Pełny eksperyment EDU 3D dla wszystkich pięciu metod:

```powershell
.\run_edu_3d_experiment.ps1 -Target DMU_01 -Mode mixed
```

Najpierw można sprawdzić generowane polecenia bez kosztownych obliczeń:

```powershell
.\run_edu_3d_experiment.ps1 -DryRun
```

Wyniki są zapisywane w `output/edu_3d_experiment_<data_i_czas>/`. Katalog `output/` pozostaje lokalny i celowo nie trafia do Git.

## Dane i sposób działania

Plik wejściowy CSV musi zawierać:

- `name` — unikalną nazwę DMU;
- `i1`, `i2`, ... — nakłady (mniej jest lepiej);
- `o1`, `o2`, ... — efekty (więcej jest lepiej).

Tryby pipeline'ów:

- `real` — ścieżka wyłącznie przez istniejące DMU;
- `fictive` — ścieżka przez wygenerowane stany osiągalne;
- `mixed` — połączenie stanów realnych i fictive.

Tryby `fictive` i `mixed` mogą być kosztowne. Liczba punktów rośnie szybko wraz z liczbą kolumn i zagęszczeniem siatki. Pierwsze próby warto wykonywać dla 1–2 zmiennych, większego `--step-pct` oraz małego `--max-candidates`.

Końcowy raport powinien zaczynać się od snapshotu bazowego liczonego na oryginalnym zbiorze referencyjnym. Java dostarcza skrajne efektywności, skrajne rangi oraz pary relacji necessary/possible; jest też ponownie uruchamiana po usunięciu kolejnych zbiorów DMU efektywnych, aby wyznaczyć fronty efektywności. Python zapisuje czytelne macierze, klasy równoważności, ranking necessary i krawędzie Hassego. Dopiero po tej sekcji prezentowane są punkty fictive, przejściowe fronty Pareto i wybrana ścieżka.

## Weryfikacja po sklonowaniu

Testy Pythona:

```powershell
python -m unittest discover -s .\python -p "test_*.py"
```

Test kompilacji Java:

```powershell
mvn -f .\java\pom.xml test
```

Test samej konfiguracji eksperymentu:

```powershell
.\run_edu_3d_experiment.ps1 -DryRun
```

## Ważne uwagi

- `output/`, `tmp/`, cache Pythona, pliki kompilacji i logi awarii JVM są ignorowane przez Git.
- Lokalne PDF-y źródłowe są także ignorowane; repozytorium można uruchomić bez nich.
- Katalog `_archive/` zawiera wcześniejsze wersje narzędzi i skrypt przywracający poprzedni układ projektu.
- Pusty `paths.csv` nie zawsze oznacza błąd: może oznaczać, że przy zadanej siatce i ograniczeniach nie istnieje monotoniczna ścieżka spełniająca kolejne progi.
- W razie błędu `Could not find artifact io.github.alabijak:dea-robustness:0.1-SNAPSHOT` należy ponownie wykonać `mvn clean install` w repozytorium `dea-robustness` przy użyciu tej samej konfiguracji Maven/JDK.
