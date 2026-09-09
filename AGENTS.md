# AGENTS.md

Ten plik dotyczy całego repozytorium. Jest instrukcją roboczą dla agentów rozwijających DEA-F. Aktualny kod i testy są źródłem prawdy; wcześniejsze wątki oraz katalog `_archive/` są wyłącznie kontekstem historycznym.

## Cel projektu

DEA-F generuje i ocenia robust improvement paths w Data Envelopment Analysis. Dla wybranej jednostki decyzyjnej (DMU) szukamy kolejnych osiągalnych stanów, które zmniejszają inputy i/lub zwiększają outputy oraz realizują cele dotyczące relacji robust DEA, rangi, efektywności albo stabilności oceny.

Python odpowiada za generowanie kandydatów, składanie ścieżek, metryki, raporty i wizualizacje. Java uruchamiana przez Maven oblicza dokładne miary robust DEA z biblioteki `io.github.alabijak:dea-robustness:0.1-SNAPSHOT`.

## Aktywna architektura

- `python/1_hasse_path_pipeline.py` — metoda necessary/Hasse.
- `python/2_front_path_pipeline.py` — historycznie nazwana metoda kolejnych warstw rankingu necessary; obecnie nie implementuje peelingu frontów efektywności.
- `python/3_best_rank_path_pipeline.py` — kamienie milowe najlepszej rangi.
- `python/4_best_efficiency_path_pipeline.py` — kamienie milowe najlepszej efektywności.
- `python/5_robustness_width_path_pipeline.py` — redukcja szerokości score albo rank.
- `python/path_pipeline_common.py` — wspólna logika siatki, osiągalności, selekcji przejść i enumeracji ścieżek.
- `python/candidate_refinement.py` — refinement, globalne próbkowanie i lokalne przeszukiwanie.
- `python/path_metrics.py` — metryki całych ścieżek.
- `python/path_selection.py` — wybór pojedynczy, leksykograficzny, Pareto i ważony.
- `python/validate_controlled_experiment.py` — audyt spójności pięciu runów kontrolowanych.
- `python/generate_five_method_robot_report.py` — samowystarczalny raport porównawczy przykładu z robotami.
- `python/minimality_certificate.py` — certyfikat minimalności względem przebadanej puli.
- `python/collect_path_metrics.py` — agregacja wyników metod.
- `python/generate_dea_input_snapshot.py` — wspólny snapshot wejścia `X/Y`, bazowych metryk robust DEA, iteracyjnych frontów efektywności, macierzy NEC/POS, rankingu necessary i diagramu Hassego.
- `python/render_*`, `python/generate_*`, `python/visualize_*` — raportowanie i wizualizacje.
- `docs/EXPERIMENT_OUTPUT_AND_VISUALIZATION.md` — kanoniczny workflow porównywalnych eksperymentów 2D i standard końcowego wykresu.
- `examples/five_unit_2d/` — pojedyncza lekka wersja referencyjna wykresu; właściwe, samowystarczalne artefakty pozostają w `output/`.
- `input/robot_technology_table1.csv` — 27 robotów z Table 1 Baker i Talluri (1997): dwa inputy (`Cost`, `Repeatability`) i dwa outputy (`Load capacity`, `Velocity`).
- `python/java_runner.py` — jedyne aktywne miejsce wywołań klas Java przez Maven.
- `java/src/main/java/org/example/` — adaptery CSV do biblioteki robust DEA.
- `input/` — dane wejściowe; `templates/` — szablony metryk i raportów.
- `run_edu_3d_experiment.ps1` — porównywalny eksperyment wszystkich pięciu metod dla dwóch inputów i jednego outputu.
- `_archive/old_items_20260531/` — kod historyczny. Nie przywracaj go do aktywnego flow bez wyraźnego polecenia.

Dokumentacja dla użytkownika znajduje się w `README.md` i `PROJECT_SUMMARY.md`. Jeśli zmienia się semantyka metody, parametrów lub wyników, zaktualizuj także te pliki i niniejszy `AGENTS.md`.

## Podstawowe pojęcia

- Inputy mają nazwy `i1`, `i2`, ... i są minimalizowane.
- Outputy mają nazwy `o1`, `o2`, ... i są maksymalizowane.
- Stan `current` jest osiągalny z `previous`, gdy żaden input nie rośnie i żaden output nie maleje (tolerancja `1e-9`).
- `real` oznacza istniejącą DMU, `fictive` — sztucznie wygenerowany stan, a `mixed` dopuszcza oba typy.
- `best_efficiency` i `worst_efficiency`: większa wartość jest lepsza.
- `best_rank` i `worst_rank`: mniejsza wartość jest lepsza.
- `score_width = best_efficiency - worst_efficiency`.
- `rank_width = worst_rank - best_rank`.
- Mniejsza szerokość oznacza stabilniejszą ocenę, ale nie musi oznaczać większej efektywności ani lepszej rangi.
- Relacja necessary oznacza przewagę dla wszystkich dopuszczalnych wag; possible — dla co najmniej jednego dopuszczalnego układu wag.

## Najważniejszy niezmiennik: ścieżka jest krokowa

Każdy etap `h` musi być rozwijany względem faktycznego poprzednika `z_(h-1)` w konkretnej gałęzi, a nie ponownie względem początkowej DMU.

Aktualny przebieg selekcji w `enumerate_state_paths()` jest następujący:

1. Weź pełną pulę punktów spełniających milestone etapu.
2. Dla konkretnego poprzednika odrzuć punkty nieosiągalne.
3. Jeśli metoda ma dodatkowy warunek przejścia, zastosuj go.
4. Policz wektor przyrostowych zmian od tego poprzednika.
5. Zostaw front Pareto minimalnych zmian osobno dla tego przejścia.
6. Oblicz `effort_from_previous`, zastosuj limit i rozwiń kolejne gałęzie.

Nie wolno przywracać globalnego przycinania kandydatów na podstawie `effort_from_start` przed enumeracją. Taki wariant może odrzucić punkt, który jest dobrym minimalnym krokiem z konkretnego poprzednika.

`--points-per-stage` zachowuje historyczną nazwę CLI, ale obecnie oznacza limit kandydatów osobno dla każdego poprzednika/przejścia (`points_per_transition`). Wartość i referencja przejścia są zapisywane w `transition_candidates.csv` oraz w kolumnach etapowych `paths.csv`.

## Koszt przejścia i normalizacja

Dla inputu przyrost poprawy to `(previous - current) / range`. Dla outputu to `(current - previous) / range`. Wszystkie czynniki mają obecnie równe wagi, więc koszt kroku jest średnią z przyrostów po aktywnych kolumnach DEA.

Zakresy normalizacji są stałe w obrębie eksperymentu i pochodzą z danych referencyjnych: `max(column) - min(column)`. Gdy zakres jest zerowy, kod używa bezpiecznego fallbacku opartego na wartości targetu, co najmniej `1.0`.

- `effort_from_previous` — koszt konkretnego przejścia; może uczestniczyć w wyborze.
- `effort_from_start` — wartość opisowa względem początkowej DMU; nie używaj jej do wyboru następnego punktu.
- `TC` — suma kosztów kolejnych przejść.
- `MSC` — największy koszt pojedynczego przejścia.

Koszty `effort_from_previous`, `effort_from_start`, `TC`, `MSC` i `cdir_endpoint` licz wyłącznie po kolumnach rzeczywiście modyfikowanych (`--columns`). Pełny zestaw inputów/outputów nadal służy do kontroli osiągalności i profilu operacyjnego. Domyślnie wagi czynników są równe; jawne `--factor-weights` normalizuj do sumy 1. Budżety `--max-stage-effort` i `--max-factor-changes` stosuj po filtrze osiągalności, ale przed wyznaczeniem frontu minimalnych zmian.

W testach i raportach sprawdzaj, że `TC` zgadza się z sumą etapowych `effort_from_previous`, a `transition_reference_name` wskazuje rzeczywistego poprzednika.

## Front minimalnych zmian

Kandydat dominuje innego kandydata dla danego poprzednika, jeżeli wymaga nie większej zmiany w każdym modyfikowanym wymiarze i ściśle mniejszej zmiany w co najmniej jednym. Front musi być liczony:

- na zmianach względem konkretnego poprzednika;
- tylko w kolumnach rzeczywiście modyfikowanych przez eksperyment;
- osobno dla każdego etapu i każdej gałęzi.

Minimalność etapu oznacza brak dominującego punktu w faktycznie ocenionej puli dla tego przejścia. Nie oznacza analitycznego minimum w całej ciągłej przestrzeni.

## Pięć metod

1. **Hasse path** — korzysta z klas równoważności relacji necessary i krawędzi pokrycia diagramu Hassego. W trybie fictive/mixed kolejne stany muszą spełniać wymagania odpowiednich klas necessary.
2. **Necessary-layer path** (`2_front_path_pipeline.py`, nazwa historyczna) — przechodzi po warstwach DAG relacji necessary; opcjonalne `--require-edge-monotonicity` respektuje krawędzie necessary. Nie utożsamiaj tych warstw z frontami efektywności DEA.
3. **Best rank path** — na etapie `h` kandydat spełnia `best_rank <= r_h`.
4. **Best efficiency path** — na etapie `h` kandydat spełnia `best_efficiency >= e_h`.
5. **Robustness width path** — kandydat spełnia `width <= w_h`; dodatkowo nie może pogorszyć głównego wskaźnika postępu: `best_efficiency` dla score width albo `best_rank` dla rank width.

Milestone’y metod 3–5 są interpolowane liniowo od wartości startowej do celu. Aktualny kod dla kandydatów realnych wymaga spełnienia bieżącego milestone’u. Nie zmieniaj tego po cichu: wariant opisany w artykule może używać poprzedniego progu i wymaga osobnej decyzji metodologicznej.

## Generowanie i ocenianie kandydatów

- Siatka bazowa jest aproksymacją przestrzeni osiągalnej. Liczba punktów rośnie wykładniczo wraz z liczbą wymiarów.
- Refinement numeryczny metod 3–5 przybliża granicę milestone’u przez iteracje między targetem a seedem.
- Lokalny search używa pudełka wokół centrum o promieniu wynikającym z rzeczywistego kroku siatki per kolumna; wspiera losowanie albo próbkowanie warstwowe.
- Metoda best efficiency może też użyć globalnego próbkowania warstwowego w całym osiągalnym zakresie.
- Ocena dużych pul jest dzielona na partie po 500 kandydatów, ponieważ duże jednorazowe wywołania solvera mogą wyczerpać pamięć JVM/SCIP.
- Dla porównywalnych eksperymentów zachowuj parametry i ziarna losowe w `experiment_params.csv`.
- Ciężkie eksperymenty uruchamiaj sekwencyjnie. Równoległe instancje Maven/JVM powodowały awarie z braku pamięci.
- Przy przygotowywaniu wspólnego outputu 2D stosuj `docs/EXPERIMENT_OUTPUT_AND_VISUALIZATION.md`. Nie łącz runów o różnych parametrach i nie zastępuj frontów liczonych względem faktycznego poprzednika frontem globalnym względem startu.
- Na wykresie 2D można pokazywać wszystkie faktycznie ocenione punkty, w tym pośrednie próby refinementu. Odróżniaj je od puli selekcyjnej: próba diagnostyczna nie staje się przez samą wizualizację kandydatem ścieżki, a front przejścia licz wyłącznie na osiągalnych punktach puli selekcyjnej spełniających milestone.
- Opcja wizualizacji `--normalize-display` skaluje każdą oś od `0%` w punkcie startowym do `100%` przy najmniejszej ocenionej wartości inputu. Traktuj ją wyłącznie jako monotoniczną transformację prezentacyjną, zachowuj oryginalne wartości w hoverze i nie używaj współrzędnych ekranowych do ponownego liczenia kosztów ani frontów.
- Przed wygenerowaniem raportu prezentacyjnego utwórz jeden `dea_input_snapshot` dla wspólnego zbioru referencyjnego. Raport ma pokazać go przed ścieżkami i jawnie rozróżniać iteracyjne fronty efektywności, ranking necessary oraz przejściowy front Pareto minimalnych zmian.
- Front efektywności licz przez peeling: przelicz `best_efficiency`, usuń DMU z wartością `1` w tolerancji, ponów obliczenie na pozostałym zbiorze. Nie wyprowadzaj frontów efektywności z macierzy NEC.
- Relacje, skrajne scores, skrajne rangi i score używane w kolejnych iteracjach peelingu są wynikami Javy. Macierze, klasy równoważności, warstwy rankingu necessary i krawędzie Hassego są strukturami wyprowadzanymi w Pythonie; nie opisuj ich wszystkich jako bezpośredniego outputu Javy.
- Do Git dodawaj tylko lekką wersję przykładową z `--plotly-mode cdn`. Nie commituj samowystarczalnego HTML z osadzonym kodem Plotly ani kopii całych katalogów wynikowych.

## Pliki wynikowe

Wspólny katalog eksperymentu powinien zawierać snapshot bazowego wejścia DEA:

- `dea_input_snapshot/decision_data.csv` — dane `X/Y` użyte przez bazowe obliczenia;
- `dea_input_snapshot/reference_dmu_results.csv` — scores, rangi, widths, front efektywności i poziom rankingu NEC dla oryginalnych DMU;
- `dea_input_snapshot/efficiency_fronts.csv` — fronty efektywności z iteracyjnego usuwania jednostek efektywnych;
- `dea_input_snapshot/necessary_matrix.csv` i `possible_matrix.csv` — macierze relacji z orientacją wiersz/source nad kolumną/target;
- `dea_input_snapshot/necessary_components.csv`, `necessary_ranking.csv` i `necessary_cover_edges.csv` — struktura rankingu i diagramu Hassego relacji necessary;

Każdy run dostaje osobny katalog `run_<timestamp>` pod katalogiem metody. Najważniejsze pliki runu:

- `fictive_candidates.csv` — wygenerowane stany przed oceną;
- `fictive_candidate_metrics.csv` — metryki robust DEA stanów;
- `stage_candidates.csv` — pełne pule spełniające milestone’y;
- `transition_candidates.csv` — branch-specific, osiągalne i niezdominowane następne kroki;
- `paths.csv` — kompletne ścieżki wraz z metadanymi etapów;
- `path_metrics.csv` — metryki całych ścieżek;
- `path_selection.csv`, `path_pareto.csv` i `path_pareto_representatives.csv` — audytowalne wybory, pełny front Pareto i reprezentanci równoważnych wektorów metryk;
- `validation_checks.csv` — kontrole spójności kontrolowanego zestawu pięciu metod;
- `all_path_metrics.csv` i `method_summary.csv` — agregaty wielu metod.

Pusty `paths.csv` może być poprawnym wynikiem: oznacza, że ocenione pule nie tworzą pełnej, osiągalnej ścieżki przez wszystkie milestone’y. Nie maskuj tego sztucznym rozluźnieniem warunków.

## Ograniczenia i ostrożne wnioskowanie

Nie opisuj obecnej implementacji jako w pełni zgodnej matematycznie z artykułem.

- Punkty fictive powstają przez grid, próbkowanie i refinement, a nie przez pełne ciągłe zadanie optymalizacyjne.
- Certyfikat minimalności dotyczy wyłącznie faktycznie przebadanej puli.
- Przycinanie do frontu Pareto i limity przejść są heurystyką; mogą usunąć lokalnie droższy krok prowadzący do ścieżki lepszej według innej metryki globalnej.
- `cdir_endpoint` jest bezpośrednim kosztem start → koniec konkretnej ścieżki. `cdir` jest wspólnym minimum względem ocenionej końcowej puli milestone'u (metody 3–5) albo końców kompletnych ścieżek (metody 1–2), a `cdir_scope` zapisuje zakres. Nadal nie jest to analityczne minimum po całej ciągłej przestrzeni, więc `DR = TC / cdir` jest aproksymacją definicji z artykułu.
- `PC` licz tylko z rzeczywistych zbiorów peer/reference. List necessary/possible nie nazywaj peer sets; ich pomocnicza miara ciągłości jest zapisywana jako `RC`.
- Ogólne `PYV/PYM` i `APW/FW/WW` muszą odpowiadać semantyce metody, a nie pierwszej dostępnej kolumnie. Metoda rangi używa rangi i rank width; metoda efektywności score i score width; metoda width używa wskazanego rodzaju szerokości oraz odpowiadającego mu wskaźnika postępu.
- Równe wagi czynników i zakresy z danych obserwowanych są świadomymi założeniami eksperymentalnymi.
- Remisy metryk porównuj z tolerancją numeryczną; przy wyborze minimalnego `TC` używaj `MSC` jako kolejnego kryterium, zamiast polegać na różnicach float rzędu błędu maszynowego.

W raportach wyraźnie rozdzielaj: wynik algorytmiczny dla ocenionej puli, dowód/test techniczny oraz twierdzenie matematyczne.

## Środowisko i uruchomienie

Wymagane są Python 3.11+, JDK 21+, Maven i pakiety z `requirements.txt`. Biblioteka robust DEA jest zależnością `SNAPSHOT` i na nowym komputerze musi być wcześniej zainstalowana lokalnie:

```powershell
git clone https://github.com/alabijak/dea-robustness.git ..\dea-robustness
Set-Location ..\dea-robustness
mvn clean install
Set-Location ..\DEA-F
```

Uruchamiaj skrypty z głównego katalogu repozytorium. Ścieżki z odstępami przekazuj jako poprawnie cytowane argumenty; adapter `java_runner.py` obsługuje ścieżki przekazywane do Maven.

## Minimalna weryfikacja zmian

Po zmianach wspólnej logiki lub pipeline’ów uruchom co najmniej:

```powershell
python -m unittest discover -s .\python -p "test_*.py"
python -m compileall -q .\python
mvn -f .\java\pom.xml test
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_edu_3d_experiment.ps1 -DryRun
git diff --check
```

Testy w `python/test_path_pipeline_common.py` chronią kluczową semantykę: aktualny poprzednik, osobny front dla gałęzi, koszt przyrostowy, stałą normalizację i sumowanie `TC`. Jeżeli zmieniasz którąkolwiek z tych zasad, najpierw dodaj lub popraw test odtwarzający konkretny przypadek.

Pełny run robust DEA jest potrzebny przy zmianach integracji Java, formatu CSV, generowania kandydatów lub warunków metod. Zacznij od małego trybu `real` albo małej siatki, zanim uruchomisz kosztowny refinement.

## Higiena repozytorium

- Przed edycją zawsze sprawdź `git status`; zachowaj istniejące zmiany użytkownika.
- Nie commituj `output/`, `tmp/`, `java/target/`, cache Pythona, logów `hs_err_pid*.log`/`replay_pid*.log` ani lokalnych PDF-ów referencyjnych.
- Nie usuwaj wyników historycznych użytkownika tylko dlatego, że są ignorowane przez Git.
- Nie modyfikuj `_archive/` podczas zwykłego rozwoju aktywnych metod.
- Dane i raporty przeznaczone do reprodukcji mogą trafić do `input/` albo `templates/`; duże wyniki eksperymentów pozostają w `output/`.
- Po zmianie formatu CSV popraw producenta, wszystkich konsumentów, raporty, wizualizacje i testy.
- Przy zmianach naukowych zapisuj założenia jawnie. Nie przedstawiaj heurystyki jako dowodu ani wyniku z ograniczonej liczby ścieżek jako pełnej enumeracji.
