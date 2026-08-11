# Standard outputu eksperymentu i wizualizacji 2D

Ten dokument opisuje sposób przygotowania porównywalnego eksperymentu DEA-F oraz jednego, samowystarczalnego pliku HTML z interaktywnym wykresem i krótką tabelą ścieżki. Jest to standard użyty w porównaniu jednostek `T_C`, `T_B`, `T_A`, `T_D` i `M07`.

Aktualny wariant dotyczy:

- metody `best_efficiency_path`;
- dwóch modyfikowanych inputów `i1` i `i2`;
- jednego stałego outputu `o1`;
- trzech etapów poprawy;
- kandydatów fictive;
- wyboru ścieżki według minimalnego `TC`, a przy remisie według minimalnego `MSC`.

## 1. Oczekiwany rezultat

W katalogu jednego eksperymentu powinny znaleźć się osobne wyniki każdej jednostki i jeden końcowy plik HTML:

```text
output/<nazwa_eksperymentu>/
├── experiment_params.csv
├── T_A/
│   ├── all_path_metrics.csv
│   ├── method_summary.csv
│   ├── minimality_certificate.csv
│   ├── minimality_certificate.md
│   └── best_efficiency_path/run_<timestamp>/...
├── T_B/
├── T_C/
├── T_D/
├── M07/
└── minimalne_sciezki_5_jednostek.html
```

Plik `minimalne_sciezki_5_jednostek.html` jest samowystarczalny: kod Plotly, dane pięciu eksperymentów, wykres i tabela są zapisane w jednym pliku. Można go otworzyć lokalnie i wysłać bez dołączania katalogów z CSV.

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

W przykładzie publikacyjnym wybrano pięć najmniej efektywnych jednostek: `T_C`, `T_B`, `T_A`, `T_D` i `M07`.

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
| `max_candidates` | `200` |
| `points_per_stage` | `20` na każdego poprzednika/przejście |
| Refinement | włączony |
| Iteracje refinement | `5` |
| Maksymalna liczba seedów | `5` |
| Lokalne próbki na centrum | `100`, grid `10×10` |
| Lokalny mnożnik kroku | `1.5` |
| Lokalna strategia | `stratified` |
| Lokalne ziarno losowe | `42` |
| Globalne próbki | `400`, grid `20×20` |
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

## 4. Uruchomienie eksperymentów

Uruchamiaj ciężkie przebiegi sekwencyjnie. Równoległe instancje Maven/JVM mogą wyczerpać pamięć i pozostawić niepełne pliki CSV.

Przykład PowerShell:

```powershell
$Root = ".\output\five_units_2d_<timestamp>"
$Targets = @("T_C", "T_B", "T_A", "T_D", "M07")

foreach ($Target in $Targets) {
  python .\python\4_best_efficiency_path_pipeline.py `
    --input .\input\publication_example_search.csv `
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
    --max-candidates 200 `
    --points-per-stage 20 `
    --refine-fictive-candidates `
    --refine-iterations 5 `
    --refine-max-seeds 5 `
    --local-search-samples 100 `
    --local-search-step-multiplier 1.5 `
    --local-search-random-state 42 `
    --local-search-sampling stratified `
    --global-search-samples 400 `
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

## 5. Agregacja i certyfikat minimalności

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

Automatyczny wybór ścieżki przebiega następująco:

1. odrzuć ścieżki z naruszeniami osiągalności;
2. znajdź minimalne `TC` z tolerancją `1e-9`;
3. wśród remisów wybierz minimalne `MSC`;
4. przy dalszym remisie użyj stabilnego porządku `path_id`.

## 6. Generowanie jednego pliku HTML

Po przygotowaniu metryk i certyfikatów uruchom:

```powershell
python .\python\generate_five_unit_2d_html.py `
  --root $Root `
  --targets T_C,T_B,T_A,T_D,M07 `
  --input .\input\publication_example_search.csv `
  --output "$Root\minimalne_sciezki_5_jednostek.html"
```

Domyślny `--plotly-mode embedded` tworzy samowystarczalny plik przeznaczony do wysłania. Lekki przykład przechowywany w repo generuj przez:

```powershell
python .\python\generate_five_unit_2d_html.py `
  --root $Root `
  --targets T_C,T_B,T_A,T_D,M07 `
  --input .\input\publication_example_search.csv `
  --plotly-mode cdn `
  --output .\examples\five_unit_2d\minimalne_sciezki_5_jednostek.html
```

Wersja `cdn` wymaga połączenia z Internetem przy otwieraniu. Nie używaj jej jako pliku wysyłanego do odbiorcy, który może pracować offline.

Generator przed zapisaniem pliku sprawdza dla każdej wybranej ścieżki:

- brak naruszeń osiągalności;
- `pareto_minimal = True` na każdym etapie;
- `dominating_points = 0` na każdym etapie.

Aktualny generator zakłada dokładnie dwa zmieniane inputy (`i1`, `i2`) i trzy etapy. Zmiana liczby wymiarów, nazw osi lub liczby etapów wymaga aktualizacji generatora i testów wizualizacji.

## 7. Standard wykresu prezentacyjnego

Wykres dla promotora lub publikacji powinien być czysty i pokazywać tylko elementy potrzebne do oceny ścieżki.

### Obowiązkowe elementy

- oś X: `i1`;
- oś Y: `i2`;
- jednakowa skala osi (`1` jednostka na X odpowiada `1` jednostce na Y);
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

## 8. Krótka tabela pod wykresem

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

## 9. Obowiązkowa kontrola poprawności

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

Minimalna weryfikacja kodu:

```powershell
python -m unittest discover -s .\python -p "test_*.py"
python -m compileall -q .\python
mvn -f .\java\pom.xml test
git diff --check
```

## 10. Jak opisywać wynik

Poprawne sformułowanie:

> Wybrany punkt każdego etapu jest niezdominowany względem faktycznego poprzednika wśród kandydatów ocenionych w tym eksperymencie.

Niepoprawne, zbyt mocne sformułowanie:

> Wykres dowodzi globalnej minimalności ścieżki w całej ciągłej przestrzeni.

Grid, refinement i próbkowanie dają skończoną pulę. Certyfikat minimalności dotyczy tej puli, a nie wszystkich możliwych punktów ciągłych. Ponadto obecne `DR` korzysta z `cdir` liczonego dla końca konkretnej ścieżki i nie jest jeszcze pełną implementacją definicji `Cdir` z artykułu.

## 11. Dwa rodzaje wizualizacji

Utrzymuj rozdział między dwoma widokami:

1. **Widok diagnostyczny** — pełne pule, punkty poniżej progu, nieosiągalne, zdominowane, front i poprzednik. Służy do sprawdzania algorytmu.
2. **Widok prezentacyjny** — DMU referencyjne, trzy fronty, minimalna ścieżka, animacja i krótka tabela. Służy do rozmowy z promotorem i prezentacji wyników.

Nie usuwaj danych diagnostycznych z runu tylko dlatego, że nie są pokazywane w końcowym HTML.
