# Standard outputu eksperymentu i wizualizacji 2D

## Porównanie dwóch celów POS na grafie NEC

### Wybor metryki w raporcie pieciu metod

Kazda metoda ma selektor 20 pozycji katalogu. Zmieniaj razem sciezke, fronty
etapow od faktycznego poprzednika, tabele stanow oraz 20 wynikow z rachunkami.
Kierunki: katalog `direction`; PYM i RR maksymalizowane, pozostale minimalizowane.
Wyklucz NaN/inf, podaj liczbe wykluczonych wynikow. Przy remisie w tolerancji 1e-9
wybierz TC, potem MSC, potem stabilne ID; pokaz tie_count i stale metryki.
Eksportuj selection/ties/transition_audit z prefiksem `metric_explorer_` przy runie,
a wspolne zestawienie wyborow w katalogu eksperymentu. Wszystkie rownorzedne ID
sa w CSV, panel pokazuje jednego reprezentanta. Zestawienie/wykres porownawczy
na gorze raportu zachowuje podpisany wybor leksykograficzny. Nie nazywaj skonczonej
puli po przycinaniu i limitach wszystkimi mozliwymi sciezkami problemu ciaglego.

Aktualny wspolny cel dla pieciu metod: uruchom `run_shared_pool_methods.py`
z `--final-goal joint --pool <porownanie> --output-root <nowy-katalog>`.
Raport `five_method_report.html` pokazuje piec metod, wszystkie z joint na koncu.
Hasse ma POS na etapach posrednich. Metoda 2 to nadal warstwy NEC, nie fronty
efektywnosci. Dla metod 3-5 joint nie zastepuje koncowego milestone'u, lecz jest
dodatkowym warunkiem. Nie porownuj Cdir ze starymi pulami bez joint. Audyt
`common_weight_final_goal` sprawdza kazdy zapisany koniec; `final_goal.json`
wskazuje cele oraz katalog swiadectw. Brak sciezki pozostaw jako jawny wynik.

Pelny raport siedmiu wariantow: po porownaniu POS uruchom
`python python/run_shared_pool_methods.py --pool output/NAZWA_POROWNANIA`.
Pieciu historycznych metod nie przedstawiaj jako poprawionych celow POS.
Wszystkie metody wykorzystuja jedna pule Java i osobno buduja sciezki.
Nie dodawaj method-specific refinementu do tej kontrolowanej wersji.
Przy `--grid-levels 9 --jitter-samples 4 --jitter-fraction 0.6` pula ma 3645 punktow.
Zestawienie zachowuje dane DEA, NEC/POS, peeling, diagram Hassego, metryki,
okienka rachunkow, etapy, pochodzenie punktow i audyt minimalnosci.

Gesty wariant: dodaj `--grid-levels 9 --jitter-samples 2 --jitter-fraction 0.2
--sampling-seed 20261019` do polecenia porownania. Dla obecnego EDU: 2187 punktow
(729 siatki + 1458 prob w lokalnych pudelkach). Obie wersje uzywaja identycznej
puli, wszystkie punkty przechodza ocene Java po maksymalnie 500 stanow w partii.
`candidate_sampling.csv` zapisuje centra i pochodzenie, `experiment_params.csv`
parametry losowania. Szum nie zmienia scores ani danych referencyjnych.

Uruchom `compare_hasse_pos_variants.py --source <zapisany_eksperyment> --output <nowy_katalog>`.
Ten workflow porównuje `pairwise` i `joint`, a nie pięć dawnych metod. Nie używa
warstw NEC. Ścieżki wynikają z grafu Hassego; pośredni cel jest pojedynczą DMU
osiąganą przez POS, a wspólny koniec obejmuje wszystkie maksymalne klasy NEC.
Zachowaj oryginalne dane, siatkę, zakresy, wagi i limity między wariantami.

Raport `hasse_pos_comparison.html` pokazuje snapshot wejścia przed ścieżkami,
kolorem rozróżnia punkty spełniające tylko POS parami od spełniających wspólne wagi,
zawiera porównanie wybranych ścieżek, etapowe fronty względem poprzednika i wszystkie
metryki. Okienka liczą rachunki z nowych wyników, nie z dawnego eksperymentu NEC.
Tabele wag pokazują osobne świadectwa dla kolejnych celów oraz jeden wspólny wektor
i ilorazy przy nim. `joint_top=false` oznacza, że wyeksportowane wagi optymalne nie
są świadectwem spełnienia celu. Zera kosztu ujawniaj, nie usuwaj ich dla estetyki.

Pule rozdzielaj przez `milestone_route_id`, `milestone_target_name` i
`milestone_goal_kind`, nie tylko numer etapu. Końce różnych gałęzi mają identyczny
zbiór celów. Pełne wyniki zachowaj w nowych katalogach `pairwise/` i `joint/`,
z `source_manifest.csv`, `graph_routes.json`, `top_targets.csv`, `possible_goals.csv`,
`possible_goals.csv.witnesses.csv`, `joint_weight_audit.csv`, `variant_summary.csv`
i `validation_checks.csv`. Nie utożsamiaj obu kwantyfikatorów ani wyniku z globalną
minimalnością w przestrzeni ciągłej. Wspólny LP dotyczy dodatnich danych i
nieograniczonych nieujemnych wag CCR; nie dowodzi silnej efektywności Pareto.

## Wariant EDU: trzy inputy, pięć metod

Dla `input/EDU.csv` używamy pełnego modelu `i1,i2,i3,o1`. Wszystkie inputy mogą maleć,
a output wybranej DMU pozostaje stały. Przykład:

```powershell
python python/run_five_method_experiment.py --input input/EDU.csv --root output/edu_five_methods_NEW --target DMU_03 --columns i1,i2,i3 --step-pct 25 --local-samples 64 --global-samples 216 --target-width 0.15 --max-paths 20000
```

Runner tworzy snapshot i uruchamia metody kolejno. Zapisuje hash wejścia, parametry,
pełne listy argumentów oraz czas każdego kroku. Generator `generate_five_method_report.py`
tworzy `five_method_report.html` i `README_results.md`. HTML zawiera wszystkie dane
i Plotly, działa offline i pokazuje wejście DEA przed wynikami pięciu testów.

- Siatka bazowa: 5 x 5 x 5 = 125 punktów; wspólna dla pięciu metod.
- Metody 3–5: 4 iteracje refinementu, maksymalnie 6 centrów na etap, 64 próbki lokalne
  na centrum, ziarno bazowe 42 (pipeline dodaje indeks etapu).
- Metoda 4 dodatkowo: 216 globalnych próbek z ziarnem 142.
- Numeryczne cele: najlepsza ranga 1, najlepsza efektywność 0.95, score width 0.15;
  każdy z trzech numerycznych generatorów ma 3 etapy. Długość metod NEC wynika z grafu.
- Koszty: równe wagi trzech inputów, zakresy z pełnego EDU. Limit 15 punktów na
  przejście i 20000 kompletnych ścieżek; osiągnięcie limitu ścieżek ujawnia walidacja.
- Wspólne osie 3D używają `aspectmode=data`. Rzuty 2D nie służą do oceny dominacji:
  niezdominowanie jest sprawdzane we wszystkich zmienianych inputach.
- Każdy etap pokazuje własne próby, front i rzeczywistego poprzednika. Warstwy punktów
  można włączać i wyłączać z legendy, a hover zawiera scores, rangi oraz źródło/iterację.
- Front Hassego jest odtwarzany wyłącznie dla zapisanego `milestone_component_id`.
  Stare `paths.csv` bez metadanych celu nie są wystarczające do tego audytu.
- Rozróżniaj liczbę rekordów ścieżek od liczby różnych sekwencji nazw stanów.
  Różne cele strukturalne mogą dawać identyczną sekwencję punktów; osobno pokazuj
  reprezentantów równoważnych wektorów metryk na froncie Pareto ścieżek.

Kontrole: `validation_checks.csv` dla wszystkich ścieżek i
`selected_transition_audit.csv` dla etapów pięciu wybranych ścieżek. Nie nazywaj
warstw NEC frontami efektywności. Nie przedstawiaj porównania jako eksperymentu
na identycznych końcowych pulach: dodatkowe próbkowanie jest zależne od metody.

Ten dokument opisuje sposób przygotowania porównywalnego eksperymentu DEA-F oraz jednego, samowystarczalnego pliku HTML z interaktywnym wykresem i krótką tabelą ścieżki. Referencyjny przykład korzysta z robotów `Robot_12`, `Robot_18`, `Robot_09`, `Robot_17` i `Robot_06` z Table 1 artykułu Baker i Talluri (1997).

Do porównania **wszystkich pięciu metod** na jednej jednostce służy osobny, kontrolowany runner:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_robot_2d_five_method_experiment.ps1 -RefineFictiveCandidates
```

Domyślnie wybiera `Robot_18`, modyfikuje `i1,i2`, utrzymuje `o1,o2` na poziomie startowym, tworzy wspólny snapshot, uruchamia metody sekwencyjnie i zapisuje parametry, ziarna, wagi oraz budżety. Folder `front_path` zachowuje historyczną nazwę techniczną; w tekście publikacji tę metodę należy nazywać **necessary-layer path**, ponieważ jej warstwy pochodzą z DAG relacji necessary, a nie z peelingu frontów efektywności DEA.

Aktualny wariant dotyczy:

- metody `best_efficiency_path`;
- dwóch modyfikowanych inputów `i1` i `i2`;
- dwóch stałych outputów: `o1` (load capacity) i `o2` (velocity);
- trzech etapów poprawy;
- kandydatów fictive;
- wyboru ścieżki według minimalnego `TC`, a przy remisie według minimalnego `MSC`.

## 1. Oczekiwany rezultat

W katalogu jednego eksperymentu powinny znaleźć się osobne wyniki każdej jednostki i jeden końcowy plik HTML:

```text
output/<nazwa_eksperymentu>/
├── experiment_params.csv
├── dea_input_snapshot/
│   ├── decision_data.csv
│   ├── reference_dmu_results.csv
│   ├── efficiency_fronts.csv
│   ├── necessary_matrix.csv
│   ├── possible_matrix.csv
│   ├── necessary_components.csv
│   ├── necessary_ranking.csv
│   ├── necessary_cover_edges.csv
│   └── snapshot_metadata.csv
├── Robot_12/
│   ├── all_path_metrics.csv
│   ├── method_summary.csv
│   ├── minimality_certificate.csv
│   ├── minimality_certificate.md
│   └── best_efficiency_path/run_<timestamp>/...
├── Robot_18/
├── Robot_09/
├── Robot_17/
├── Robot_06/
└── minimalne_sciezki_5_jednostek.html
```

Plik `minimalne_sciezki_5_jednostek.html` jest samowystarczalny: kod Plotly, snapshot bazowego wejścia DEA, dane pięciu eksperymentów, wykres i tabele są zapisane w jednym pliku. Można go otworzyć lokalnie i wysłać bez dołączania katalogów z CSV.

Lekka wersja referencyjna znajduje się w `examples/five_unit_2d/minimalne_sciezki_5_jednostek.html`. Ma te same dane i interakcje, ale pobiera Plotly z CDN, aby nie dodawać do Git około 4,9 MB wygenerowanego kodu biblioteki.

Dokładne parametry tej wersji są zapisane w `examples/five_unit_2d/experiment_params.csv`.

Duże katalogi `output/` są lokalne i nie powinny trafiać do Git. Do repozytorium trafiają generator, dokumentacja, testy i ewentualnie małe dane wejściowe potrzebne do reprodukcji.

## 2. Wybór jednostek

Do porównania wybieraj jednostki, które:

1. mają efektywność początkową niższą od celu;
2. dają niezerową ścieżkę poprawy;
3. reprezentują różne położenia w przestrzeni `i1`–`i2`;
4. są liczone na tym samym zbiorze referencyjnym;
5. mogą być ocenione z identycznymi parametrami eksperymentu.

W przykładzie publikacyjnym wybrano pięć jednostek o najniższym początkowym `best_efficiency`: `Robot_12`, `Robot_18`, `Robot_09`, `Robot_17` i `Robot_06`.

Nie łącz w jednym wykresie wyników pochodzących z różnych wersji algorytmu, innych zakresów normalizacji albo różnych gęstości próbkowania bez wyraźnego oznaczenia tej różnicy.

## 3. Wspólne parametry przykładu

W porównaniu pięciu jednostek użyto następującej konfiguracji:

| Parametr | Wartość |
|---|---:|
| Metoda | `best_efficiency_path` |
| Tryb | `fictive` |
| Modyfikowane kolumny | `i1,i2` |
| Docelowa najlepsza efektywność | `0.92` |
| Liczba etapów | `3` |
| `pct_above` | `5` |
| `step_pct` | `25` |
| `min_points_per_dim` | `5` |
| `max_candidates` | `100` |
| `points_per_stage` | `15` na każdego poprzednika/przejście |
| Refinement | włączony |
| Iteracje refinement | `3` |
| Maksymalna liczba seedów | `3` |
| Lokalne próbki na centrum | `25`, grid `5×5` |
| Lokalny mnożnik kroku | `1.5` |
| Lokalna strategia | `stratified` |
| Lokalne ziarno losowe | `42` |
| Globalne próbki | `100`, grid `10×10` |
| Globalne ziarno losowe | `142` |

`--points-per-stage` jest historyczną nazwą parametru. W aktualnym kodzie oznacza limit niezdominowanych kandydatów osobno dla każdego faktycznego poprzednika.

Zapisz wspólną konfigurację w `experiment_params.csv` w głównym katalogu eksperymentu. Plik powinien zawierać co najmniej:

- ścieżkę i sumę kontrolną albo wersję danych wejściowych;
- listę targetów w kolejności prezentacji;
- nazwę pipeline'u i wszystkie przekazane argumenty CLI;
- ziarna losowe;
- zakresy normalizacji;
- znaczenie `points_per_stage` jako limitu na przejście;
- identyfikator commita Git albo informację o lokalnych zmianach;
- datę rozpoczęcia i zakończenia;
- status każdego przebiegu.

Nie zapisuj tylko parametrów różniących się od wartości domyślnych. Raport ma pozwalać odtworzyć pełne polecenie bez zgadywania wersji domyślnych.

## 4. Bazowy snapshot wejścia DEA

Przed uruchomieniem prezentacji wygeneruj jeden wspólny snapshot dla dokładnie tego samego pliku wejściowego, który jest używany przez wszystkie porównywane jednostki:

```powershell
python .\python\generate_dea_input_snapshot.py `
  --input .\input\robot_technology_table1.csv `
  --output-dir "$Root\dea_input_snapshot" `
  --java-entry .\java
```

Etap uruchamia sekwencyjnie adaptery Javy:

- `CsvPreferenceRelationsPreview` — pełne relacje necessary i possible;
- `CsvExtremeEfficienciesExporter` — `best_efficiency` i `worst_efficiency`;
- `CsvExtremeRanksExporter` — `best_rank` i `worst_rank`.

Następnie `CsvExtremeEfficienciesExporter` jest uruchamiany ponownie na coraz mniejszym zbiorze: jednostki z `best_efficiency = 1` w tolerancji trafiają do bieżącego frontu efektywności i są usuwane przed następną iteracją.

Następnie Python zapisuje:

- macierz decyzyjną `X/Y` w `decision_data.csv`;
- połączoną tabelę DMU, scores, rang, widths, frontów efektywności i poziomów rankingu NEC w `reference_dmu_results.csv`;
- iteracyjne fronty efektywności w `efficiency_fronts.csv`;
- macierze `necessary_matrix.csv` i `possible_matrix.csv`, gdzie wiersz/source jest preferowany nad kolumną/target;
- klasy równoważności, ranking necessary i krawędzie Hassego.

W raporcie prezentuj ten snapshot przed ścieżkami. `efficiency_front = 1` zawiera jednostki efektywne w pierwszym przeliczeniu. `necessary_rank = 1` zawiera maksymalne klasy częściowego porządku necessary. Front efektywności, ranking necessary i front Pareto minimalnych zmian dla przejścia ścieżki są trzema różnymi obiektami.

## 5. Uruchomienie eksperymentów

Uruchamiaj ciężkie przebiegi sekwencyjnie. Równoległe instancje Maven/JVM mogą wyczerpać pamięć i pozostawić niepełne pliki CSV.

Przykład PowerShell:

```powershell
$Root = ".\output\five_units_2d_<timestamp>"
$Targets = @("Robot_12", "Robot_18", "Robot_09", "Robot_17", "Robot_06")

foreach ($Target in $Targets) {
  python .\python\4_best_efficiency_path_pipeline.py `
    --input .\input\robot_technology_table1.csv `
    --target $Target `
    --target-best-efficiency 0.92 `
    --stages 3 `
    --output-dir "$Root\$Target" `
    --java-entry .\java `
    --mode fictive `
    --columns i1,i2 `
    --pct-above 5 `
    --step-pct 25 `
    --min-points-per-dim 5 `
    --max-candidates 100 `
    --points-per-stage 15 `
    --refine-fictive-candidates `
    --refine-iterations 3 `
    --refine-max-seeds 3 `
    --local-search-samples 25 `
    --local-search-step-multiplier 1.5 `
    --local-search-random-state 42 `
    --local-search-sampling stratified `
    --global-search-samples 100 `
    --global-search-random-state 142

  if ($LASTEXITCODE -ne 0) {
    throw "Eksperyment dla $Target zakończył się błędem."
  }
}
```

Każdy przebieg tworzy własny `run_<timestamp>`. Do dalszych kroków używaj wyłącznie kompletnego runu zawierającego co najmniej:

- `stage_candidates.csv`;
- `transition_candidates.csv`;
- `paths.csv`;
- `path_metrics.csv`.

## 6. Agregacja i certyfikat minimalności

Dla każdej jednostki agreguj metryki i twórz certyfikat względem ocenionej puli:

```powershell
foreach ($Target in $Targets) {
  python .\python\collect_path_metrics.py `
    --experiment-dir "$Root\$Target"

  python .\python\minimality_certificate.py `
    --experiment-dir "$Root\$Target" `
    --columns i1,i2 `
    --output-md "$Root\$Target\minimality_certificate.md" `
    --output-csv "$Root\$Target\minimality_certificate.csv"

  if ($LASTEXITCODE -ne 0) {
    throw "Walidacja dla $Target zakończyła się błędem."
  }
}
```

Certyfikat minimalności zachowuje historyczny wybór kontrolny:

1. odrzuć ścieżki z naruszeniami osiągalności;
2. znajdź minimalne `TC` z tolerancją `1e-9`;
3. wśród remisów wybierz minimalne `MSC`;
4. przy dalszym remisie użyj stabilnego porządku `path_id`.

Do wyboru ścieżki pokazywanej w pracy używaj dodatkowych plików tworzonych przez każdy pipeline:

- `path_selection.csv` — minimum pojedynczych kryteriów, wybór leksykograficzny i znormalizowany wybór ważony;
- `path_pareto.csv` — niezdominowane ścieżki w przestrzeni kryteriów.
- `path_pareto_representatives.csv` — po jednym reprezentancie dla identycznych (z tolerancją zapisu) wektorów kryteriów, co ogranicza wizualne duplikaty bez ukrywania pełnego frontu.

Dla metod best-rank, best-efficiency i width domyślny wybór leksykograficzny minimalizuje kolejno `MD`, `MSC`, `SWBP`, `FW`. Dla metod Hasse i necessary-layer, które nie mają numerycznego odchylenia milestone'u, porządek to `MSC`, `TC`, `SWBP`, `FW`. W raporcie pokazuj też zwycięzcę `min TC`, jeśli różni się od wyboru publikacyjnego — taki kontrast ujawnia koszt ścisłego śledzenia kamieni milowych.

## 7. Generowanie jednego pliku HTML

Po przygotowaniu metryk i certyfikatów uruchom:

```powershell
python .\python\generate_five_unit_2d_html.py `
  --root $Root `
  --targets Robot_12,Robot_18,Robot_09,Robot_17,Robot_06 `
  --input .\input\robot_technology_table1.csv `
  --x-label "Koszt (`$10 000)" `
  --y-label "Powtarzalność (mm)" `
  --independent-axis-scales `
  --output "$Root\minimalne_sciezki_5_jednostek.html"
```

Domyślny `--plotly-mode embedded` tworzy samowystarczalny plik przeznaczony do wysłania. Lekki przykład przechowywany w repo generuj przez:

```powershell
python .\python\generate_five_unit_2d_html.py `
  --root $Root `
  --targets Robot_12,Robot_18,Robot_09,Robot_17,Robot_06 `
  --input .\input\robot_technology_table1.csv `
  --x-label "Koszt (`$10 000)" `
  --y-label "Powtarzalność (mm)" `
  --independent-axis-scales `
  --plotly-mode cdn `
  --output .\examples\five_unit_2d\minimalne_sciezki_5_jednostek.html
```

Wersja `cdn` wymaga połączenia z Internetem przy otwieraniu. Nie używaj jej jako pliku wysyłanego do odbiorcy, który może pracować offline.

Generator domyślnie odczytuje snapshot z `$Root\dea_input_snapshot`. Dla innego położenia podaj `--dea-snapshot-dir <katalog>`. Brak snapshotu albo inna kolejność DMU niż w `--input` przerywa generowanie raportu, aby nie połączyć ścieżek z niezgodnym wynikiem bazowym.

Generator przed zapisaniem pliku sprawdza dla każdej wybranej ścieżki:

- brak naruszeń osiągalności;
- `pareto_minimal = True` na każdym etapie;
- `dominating_points = 0` na każdym etapie.

`minimality_certificate.csv` zapisuje również wszystkie kolumny outputów obecne w ścieżce, a raport Markdown wymienia ich wartości początkowe i potwierdza, czy pozostały stałe na każdym etapie.

Interaktywny wykres pokazuje dla wybranego etapu wszystkie faktycznie ocenione punkty z siatki bazowej, próbkowania globalnego, iteracji refinementu, końcowej granicy refinementu i przeszukiwania lokalnego. Hover podaje `best_efficiency`, `worst_efficiency`, `score_width`, przedział rang, źródło punktu i przynależność do puli selekcyjnej. Punkty są rozdzielone na: poniżej progu, próby diagnostyczne poza pulą selekcyjną, nieosiągalne z poprzednika, zdominowane oraz front Pareto konkretnego przejścia. W raporcie pieciu metod wszystkie kategorie wybranego etapu i oryginalne DMU sa domyslnie widoczne. Kolory sa normalizowane do `0–1` wedlug minimum i maksimum E* bazowej ocenionej puli danej metody, stale dla etapow i metryk wyboru. Legenda oraz hover zachowuja rzeczywiste E*. Normalizacja jest wylacznie prezentacyjna i nie zmienia zadnej metryki ani selekcji sciezek.

Dla wymiarów o różnych rzędach wielkości można użyć `--normalize-display`. Wtedy każda oś pokazuje liniowo przeskalowany poziom redukcji: `0%` odpowiada wartości startowej targetu, a `100%` najmniejszej wartości faktycznie ocenionej w danym inpucie. Jest to wyłącznie transformacja prezentacyjna; oryginalne wartości pozostają w hoverze i tabeli, a monotoniczne skalowanie nie zmienia osiągalności, dominacji ani składu frontu Pareto. Parametr i semantykę skali zapisz w `experiment_params.csv`.

Aktualny generator zakłada dokładnie dwa zmieniane inputy (`i1`, `i2`) i trzy etapy. Zmiana liczby wymiarów, nazw osi lub liczby etapów wymaga aktualizacji generatora i testów wizualizacji.

## 8. Standard raportu i wykresu prezentacyjnego

Raport dla promotora lub publikacji powinien najpierw pokazać stan wejściowy DEA, a następnie czysty wykres zawierający tylko elementy potrzebne do oceny ścieżki.

### Obowiązkowa sekcja wejścia DEA

Przed wykresem ścieżki pokaż:

- wszystkie oryginalne DMU i macierz decyzyjną `X/Y`;
- `best_efficiency`, `worst_efficiency` i `score_width`;
- `best_rank`, `worst_rank` i `rank_width`;
- front efektywności oraz klasę i poziom rankingu necessary każdej DMU;
- podsumowanie wybranej DMU, w tym jednostki NEC nad nią i pod nią;
- pełne macierze necessary i possible z opisaną orientacją;
- klasy/ranking necessary i graficzny diagram Hassego z wyróżnionymi ścieżkami od wybranej DMU.

Macierze i struktury szczegółowe mogą być domyślnie zwinięte, ale muszą pozostać dostępne w tym samym pliku HTML. Zmiana wybranej DMU ma aktualizować jej podsumowanie, wyróżnienie w tabeli i zaznaczenie wiersza/kolumny macierzy.

### Obowiązkowe elementy

- oś X: `i1`;
- oś Y: `i2`;
- jawnie opisana skala każdej osi; dla robotów używamy niezależnych skal, ponieważ koszt i powtarzalność mają różne jednostki oraz zakresy;
- te same zakresy osi dla wszystkich porównywanych DMU;
- jednostki referencyjne jako małe, półprzezroczyste, szare punkty;
- osobny front Pareto minimalnych zmian dla każdego etapu;
- wyraźna linia łącząca rzeczywiste kolejne punkty ścieżki;
- bezpośrednie etykiety `Start`, `1`, `2`, `3`;
- hover z `i1`, `i2`, efektywnością i nazwą typu punktu;
- legenda rozróżniająca jednostki referencyjne, fronty i wybrane punkty;
- możliwość przesuwania, przybliżania i resetowania widoku;
- animacja `Start → etap 1 → etap 2 → etap 3`;
- przełącznik jednostki bez przeładowywania pliku.

### Kolory i symbole

| Element | Kolor | Symbol |
|---|---|---|
| Jednostki referencyjne | szary `#8a99a8` | małe koło |
| Start | pomarańczowy `#f0aa18` | romb |
| Front i punkt etapu 1 | czerwony `#df5a56` | koło |
| Front i punkt etapu 2 | niebieski `#3978b5` | kwadrat |
| Front i punkt etapu 3 | zielony `#43a06b` | romb |
| Linia ścieżki | granatowy `#17324a` | ciągła linia |

Kolor nie może być jedynym nośnikiem informacji. Etapy różnią się także symbolem i etykietą liczbową.

### Czego nie umieszczać w wersji prezentacyjnej

- pełnej chmury wszystkich zdominowanych kandydatów;
- technicznych nazw typu `stage_02_eff_local_0001_172`;
- logów przebiegu i parametrów JVM;
- kilkudziesięciu paneli diagnostycznych;
- nieporównywalnych osi lub automatycznie różnych proporcji wykresu;
- frontu policzonego globalnie względem punktu startowego.

Pełna chmura kandydatów jest przydatna diagnostycznie. Do takiej analizy użyj `python/visualize_minimality_2d.py`, ale nie traktuj wykresu diagnostycznego jako końcowej figury prezentacyjnej.

## Pełne metryki w raporcie pięciu metod

Runner `run_five_method_experiment.py` automatycznie uzupełnia PC przez
`complete_experiment_metrics.py`, a następnie generuje raport. Dla istniejącego
eksperymentu uruchom ten etap przed ponownym `generate_five_method_report.py`.
Nie trzeba ponownie generować ani oceniać kandydatów robust DEA.

Raport ma pokazać 20 wierszy katalogu (w tym pomocnicze Cdir_endpoint i RC),
opisy przed wynikami oraz po jednej kolumnie dla wybranej ścieżki każdej metody.
CSV: `selected_path_all_metrics.csv`.
W pełnej tabeli każdy wynik udostępnia okienko z rachunkiem po najechaniu lub
uzyskaniu fokusu przez Tab. Kliknięcie przypina okienko, Esc, przycisk zamknięcia
lub kliknięcie poza nim zamyka je. Pokazuj wzór, podstawione wartości etapów,
zakresy i wagi oraz wynik przed zaokrągleniem tabeli. Dla Cdir pokaż właściwą pulę
i punkt osiągający minimum; nie zastępuj go końcem wybranej ścieżki.
`metric_calculation_details.json` zachowuje te same rachunki. Generator sprawdza
ich zgodność z CSV i przerywa generowanie przy rozbieżności. Nie przeliczaj DEA
ani ścieżek tylko w celu dodania okienek. Skrypt i style są osadzone w HTML.

Rozwijana tabela każdej ścieżki zachowuje
także szczegółowe warianty PYV/PYM dla czterech wskaźników oraz szerokości score/rank.
`metric_coverage.csv` obejmuje wszystkie ścieżki, nie tylko pięć wybranych.
`metric_exceptions.csv` odróżnia brak danych od matematycznej nieokreśloności:
zerowy koszt kroku oznacza niezdefiniowane PYV/PYM, zerowe Cdir oznacza niezdefiniowane DR.
Nigdy nie zastępuj tych wartości zerem.

Przyjęta konwencja PC: input-oriented CCR na stałych oryginalnych DMU,
minimalizacja theta, a następnie przy theta=theta* osobna maksymalizacja każdej lambda_j.
P(z) zawiera wszystkie DMU z max lambda_j > 1e-7. Nie stosujemy drugiej fazy slacków;
adapter wymaga dodatnich danych. `peer_theta` może przekroczyć 1, nie jest robust score.
Tabela etapów zawiera `peer_refs`, `peer_theta`, `peer_jaccard_distance`.
PC jest średnią odległości Jaccarda wszystkich kolejnych par. RC to osobna miara
zbiorów relacji, również liczona od startu; nie zastępuje peerów.
`peer_evaluation_results.csv` zawiera także maksymalne lambda, a
`metric_completion.json` konwencję, tolerancje, hash wejścia i czas przeliczenia.
Brakującego zbioru nie utożsamiaj ze znanym pustym zbiorem `[]`.
Nie zmieniaj wybranych ścieżek tylko po to, żeby usunąć wyjątki metryk.

## 9. Krótka tabela pod wykresem

Tabela ma mieć dokładnie jeden wiersz na stan ścieżki:

| Etap | Punkt | i1 | i2 | Efektywność | Wysiłek kroku | Status |
|---:|---|---:|---:|---:|---:|---|
| 0 | nazwa DMU | wartość | wartość | score startowy | `–` | `start` |
| 1 | punkt etapu 1 | wartość | wartość | score | `effort_from_previous` | `Pareto-min.` |
| 2 | punkt etapu 2 | wartość | wartość | score | `effort_from_previous` | `Pareto-min.` |
| 3 | punkt etapu 3 | wartość | wartość | score | `effort_from_previous` | `Pareto-min.` |

Nad tabelą pokaż:

- nazwę DMU;
- stałą wartość `o1`;
- `TC`;
- `MSC`;
- `DR`;
- liczbę sprawdzonych kompletnych ścieżek.

Nie pokazuj surowych nazw technicznych punktów w tabeli prezentacyjnej. Pełne identyfikatory pozostają w `paths.csv` i `minimality_certificate.csv`.

## 10. Obowiązkowa kontrola poprawności

Przed przekazaniem wykresu sprawdź programowo:

1. `attainable_transition_violations == 0` dla wybranej ścieżki;
2. `transition_reference_name` etapu `h` jest równy nazwie punktu z etapu `h-1`;
3. `TC == sum(stage_h_effort_from_previous)` z tolerancją `1e-9`;
4. `dominating_points == 0` dla każdego etapu;
5. `pareto_minimal == True` dla każdego etapu;
6. wszystkie jednostki używają tych samych parametrów i ziaren losowych;
7. plik HTML zawiera wszystkie wybrane jednostki i działa bez plików pomocniczych;
8. skala osi jest równa i wspólna dla jednostek;
9. tabela jest aktualizowana po zmianie DMU;
10. animacja rysuje przejścia od faktycznego poprzednika.
11. snapshot zawiera tę samą listę i kolejność DMU co plik `--input`;
12. macierze NEC/POS są kwadratowe, mają kompletne etykiety i orientację wiersz/source nad kolumną/target;
13. zmiana DMU aktualizuje podsumowanie bazowych scores, rang i frontu NEC.

Minimalna weryfikacja kodu:

```powershell
python -m unittest discover -s .\python -p "test_*.py"
python -m compileall -q .\python
mvn -f .\java\pom.xml test
git diff --check
```

## 11. Jak opisywać wynik

Poprawne sformułowanie:

> Wybrany punkt każdego etapu jest niezdominowany względem faktycznego poprzednika wśród kandydatów ocenionych w tym eksperymencie.

Niepoprawne, zbyt mocne sformułowanie:

> Wykres dowodzi globalnej minimalności ścieżki w całej ciągłej przestrzeni.

Grid, refinement i próbkowanie dają skończoną pulę. Certyfikat minimalności dotyczy tej puli, a nie wszystkich możliwych punktów ciągłych. `cdir_endpoint` opisuje bezpośredni koszt do końca konkretnej ścieżki, natomiast wspólny `cdir` jest minimum w ocenionej końcowej puli celu (lub wśród końców kompletnych ścieżek metod strukturalnych). `cdir_scope` ujawnia ten zakres. Jest to lepsza aproksymacja definicji `Cdir` z artykułu, ale nadal nie dowód minimum w całej przestrzeni ciągłej.

## 12. Dwa rodzaje wizualizacji

Utrzymuj rozdział między dwoma widokami:

1. **Widok diagnostyczny** — pełne pule, punkty poniżej progu, nieosiągalne, zdominowane, front i poprzednik. Służy do sprawdzania algorytmu.
2. **Widok prezentacyjny** — zwarty snapshot wejścia DEA, DMU referencyjne, trzy przejściowe fronty Pareto, minimalna ścieżka, animacja i krótka tabela. Służy do rozmowy z promotorem i prezentacji wyników.

Nie usuwaj danych diagnostycznych z runu tylko dlatego, że nie są pokazywane w końcowym HTML.
