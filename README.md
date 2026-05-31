# DEA-F: Robust Improvement Paths

Repo zawiera pipeline'y Python + Java do generowania robust improvement paths w DEA na podstawie:

- `_archive/old_items_20260531/Robust_Improvement_Paths_in_Data_Envelopment_Analysis.pdf`

Python orkiestruje przebieg, a Java uruchamiana przez Maven liczy dokładne metryki robust DEA z biblioteki `robustDEA`.

## Porzadek W Repo

Aktywna czesc repo zostala ograniczona do metod 1-5 i ich bezposrednich zaleznosci.
Starsze skrypty, klasy Java, wyniki, build artifacts, PDF-y referencyjne oraz lokalne kopie
`robustDEA` sa w `_archive/old_items_20260531`.

Mapowanie do poprzedniego ukladu jest w `_archive/old_items_20260531/restore_map.csv`.
Przywracanie poprzedniego ukladu:

```powershell
.\_archive\old_items_20260531\restore_archived_items.ps1
```

## Pipeline'y

1. `python/1_hasse_path_pipeline.py`
2. `python/2_front_path_pipeline.py`
3. `python/3_best_rank_path_pipeline.py`
4. `python/4_best_efficiency_path_pipeline.py`
5. `python/5_robustness_width_path_pipeline.py`

Wszystkie pipeline'y obsługują tryby:

- `--mode real` - ścieżki przez istniejące DMU.
- `--mode fictive` - ścieżki przez sztucznie generowane stany osiągalne.
- `--mode mixed` - łączy kandydatów realnych i fictive.

Tryb `fictive` generuje siatkę stanów osiągalnych od DMU startowego, ocenia każdy stan przez `robustDEA`, a potem filtruje kandydatów według milestone'ów z paperu.

## Klasy Java

Główne klasy uruchamiane przez Maven:

- `org.example.CsvPreferenceRelationsPreview`
- `org.example.CsvExtremeRanksExporter`
- `org.example.CsvExtremeEfficienciesExporter`
- `org.example.CsvCandidateRobustMetricsExporter`

Nowa klasa `CsvCandidateRobustMetricsExporter` liczy dla każdego fictive state:

- `best_efficiency`
- `worst_efficiency`
- `best_rank`
- `worst_rank`
- `score_width = best_efficiency - worst_efficiency`
- `rank_width = worst_rank - best_rank`
- relacje necessary/possible względem DMU referencyjnych

## Format Wejścia

CSV musi mieć kolumnę `name`, inputy `i1, i2, ...` i outputy `o1, o2, ...`.

Przykład:

```csv
name,i1,i2,i3,i4,o1,o2
WAW,10.5,36,129.4,7,9.5,129.7
KRK,3.1,19,31.6,7.9,2.9,31.3
```

## Start

Uruchamiaj z katalogu głównego repo:

```powershell
Set-Location "C:\Users\Jurek\Desktop\Doktorat\ROK 2\DEA-F"
```

## Parametry Fictive

Najważniejsze parametry dla trybu `fictive` i `mixed`:

- `--columns i1,o1` - modyfikowane inputy/outputy; jeśli pominiesz, użyte będą wszystkie kolumny DEA.
- `--pct-above 30` - jak daleko poza obserwowane minimum/maksimum budować siatkę.
- `--step-pct 10` - krok siatki jako procent zakresu dla każdej zmiennej.
- `--step-abs 0.1` - opcjonalny krok absolutny zamiast procentowego.
- `--min-points-per-dim 3` - minimalna liczba punktów na wymiar.
- `--max-candidates 2000` - twardy limit kandydatów, żeby nie przeciążyć komputera.
- `--points-per-stage 50` - opcjonalny limit kandydatów zostawianych na etap po filtracji milestone'u.
- `--max-paths 100` - limit zapisanych ścieżek.

Przy wielu kolumnach liczba kandydatów rośnie jak iloczyn liczby punktów po wymiarach, więc najpierw testuj na 1-2 kolumnach.

## Parametry Metody

Parametry wspólne:

- `--input` - plik CSV z danymi DMU.
- `--target` - DMU startowy, dla którego budowana jest ścieżka poprawy.
- `--output-dir` - katalog bazowy wyników; każdy run dostaje osobny folder `run_...`.
- `--java-entry` - katalog projektu Java, zwykle `.\java`.
- `--mode` - typ ścieżki: `real`, `fictive` albo `mixed`.
- `--max-paths` - maksymalna liczba ścieżek zapisana do `paths.csv`.
- `--maven-executable` - komenda Maven, domyślnie `mvn`.

Parametry etapów:

- `--stages` - liczba etapów ścieżki dla metod rank, efficiency i width.
- `--target-best-rank` - docelowa najlepsza ranga w `3_best_rank_path_pipeline.py`; niższa wartość oznacza ambitniejszy cel.
- `--target-best-efficiency` - docelowy najlepszy score w `4_best_efficiency_path_pipeline.py`; zwykle `1.0`.
- `--target-width` - docelowa szerokość robustności w `5_robustness_width_path_pipeline.py`; musi być nie większa niż aktualna szerokość targetu.
- `--width-kind` - typ szerokości w metodzie 5: `score` albo `rank`.
- `--require-edge-monotonicity` - tylko dla `2_front_path_pipeline.py`; wymusza przejścia po rzeczywistych krawędziach porządku necessary.

Największy wpływ na koszt obliczeń mają:

- `--columns`
- `--step-pct`
- `--max-candidates`
- `--points-per-stage`
- `--stages`

Mniejszy `--step-pct` daje dokładniejszą siatkę, ale szybko zwiększa liczbę kandydatów. Większe `--points-per-stage` zostawia więcej alternatyw na każdym etapie, więc może zwiększyć liczbę ścieżek kombinatorycznie.

## 1. Hasse Path

Real:

```powershell
python .\python\1_hasse_path_pipeline.py --input .\input\airports.csv --target KAT --output-dir .\output --java-entry .\java --mode real --max-paths 200
```

Mixed z fictive states:

```powershell
python .\python\1_hasse_path_pipeline.py --input .\input\airports.csv --target KAT --output-dir .\output --java-entry .\java --mode mixed --columns o1 --step-pct 50 --min-points-per-dim 3 --max-candidates 10 --points-per-stage 5 --max-paths 20
```

Najważniejsze wyniki:

- `preference_relations_all.csv`
- `necessary_matrix.csv`
- `necessary_components.csv`
- `necessary_cover_edges.csv`
- `real_paths.csv`
- `fictive_candidate_metrics.csv`
- `stage_candidates.csv`
- `paths.csv`

Interpretacja eksperymentu:

- `necessary_components.csv` pokazuje klasy równoważności relacji necessary.
- `necessary_cover_edges.csv` pokazuje krawędzie diagramu Hassego, czyli bezpośrednie robust przejścia do lepszych klas.
- `real_paths.csv` to ścieżki po istniejących DMU.
- `paths.csv` w trybie `fictive` albo `mixed` to finalne ścieżki stanów, które spełniły wymagania kolejnych klas.
- Dobra ścieżka Hassego oznacza przejście od targetu do klasy maksymalnej przez kolejne robustly lepsze milestone'y.
- Pusty `paths.csv` oznacza, że przy danej siatce i ograniczeniach nie znaleziono stanu spełniającego wymagania klas po drodze.

## 2. Front Path

Real:

```powershell
python .\python\2_front_path_pipeline.py --input .\input\airports.csv --target KAT --output-dir .\output --java-entry .\java --mode real --max-paths 200
```

Real z mocniejszą monotonicznością po krawędziach necessary:

```powershell
python .\python\2_front_path_pipeline.py --input .\input\airports.csv --target KAT --output-dir .\output --java-entry .\java --mode real --require-edge-monotonicity --max-paths 200
```

Mixed z fictive states:

```powershell
python .\python\2_front_path_pipeline.py --input .\input\airports.csv --target KAT --output-dir .\output --java-entry .\java --mode mixed --columns o1 --step-pct 50 --min-points-per-dim 3 --max-candidates 10 --points-per-stage 5 --max-paths 20
```

Najważniejsze wyniki:

- `preference_relations_all.csv`
- `fronts.csv`
- `component_paths.csv`
- `real_paths.csv`
- `fictive_candidate_metrics.csv`
- `stage_candidates.csv`
- `paths.csv`

Interpretacja eksperymentu:

- `fronts.csv` pokazuje przypisanie klas necessary do frontów.
- `front = 1` oznacza najlepszą warstwę robust frontu.
- `component_paths.csv` pokazuje przejście front po froncie od frontu targetu do `F1`.
- `--require-edge-monotonicity` zawęża ścieżki do takich, które nie tylko idą do lepszego frontu, ale też respektują krawędzie necessary.
- W trybie `fictive` milestone'em jest front, a stan sztuczny musi spełnić wymaganie robust względem klas z danego frontu.
- Pusty wynik zwykle oznacza, że siatka fictive nie zawierała punktu, który da się monotonicznie połączyć z kolejnymi frontami.

## 3. Best Rank Path

Real:

```powershell
python .\python\3_best_rank_path_pipeline.py --input .\input\airports.csv --target KAT --target-best-rank 1 --stages 3 --output-dir .\output --java-entry .\java --mode real --max-paths 200
```

Fictive:

```powershell
python .\python\3_best_rank_path_pipeline.py --input .\input\airports.csv --target KAT --target-best-rank 1 --stages 3 --output-dir .\output --java-entry .\java --mode fictive --columns o1 --step-pct 50 --min-points-per-dim 3 --max-candidates 10 --points-per-stage 5 --max-paths 20
```

Warunek fictive z paperu:

- na etapie `h`: `best_rank(z_h) <= r_h`

Najważniejsze wyniki:

- `extreme_ranks.csv`
- `rank_metrics.csv`
- `rank_milestones.csv`
- `fictive_candidate_metrics.csv`
- `stage_candidates.csv`
- `paths.csv`

Interpretacja eksperymentu:

- `rank_metrics.csv` pokazuje `best_rank` i `worst_rank` dla istniejących DMU.
- `rank_milestones.csv` pokazuje sekwencję `r_h`, czyli oczekiwane progi najlepszej rangi na kolejnych etapach.
- W trybie `real` kandydaci etapowi są istniejącymi DMU najbliższymi danemu milestone'owi.
- W trybie `fictive` kandydat etapu `h` musi spełnić `best_rank <= r_h`.
- `milestone_gap` mówi, jak daleko kandydat jest od idealnego milestone'u; mniejsze wartości są lepsze.
- `paths.csv` pokazuje konkretne ścieżki stanów od targetu do docelowej rangi.

## 4. Best Efficiency Path

Real:

```powershell
python .\python\4_best_efficiency_path_pipeline.py --input .\input\airports.csv --target KAT --target-best-efficiency 1.0 --stages 3 --output-dir .\output --java-entry .\java --mode real --max-paths 200
```

Fictive:

```powershell
python .\python\4_best_efficiency_path_pipeline.py --input .\input\airports.csv --target KAT --target-best-efficiency 1.0 --stages 3 --output-dir .\output --java-entry .\java --mode fictive --columns o1 --step-pct 50 --min-points-per-dim 3 --max-candidates 10 --points-per-stage 5 --max-paths 20
```

Warunek fictive z paperu:

- na etapie `h`: `best_efficiency(z_h) >= e_h`

Najważniejsze wyniki:

- `extreme_efficiencies.csv`
- `efficiency_metrics.csv`
- `efficiency_milestones.csv`
- `fictive_candidate_metrics.csv`
- `stage_candidates.csv`
- `paths.csv`

Interpretacja eksperymentu:

- `efficiency_metrics.csv` pokazuje `best_efficiency`, `worst_efficiency` i szerokość score dla istniejących DMU.
- `efficiency_milestones.csv` pokazuje sekwencję `e_h`, czyli oczekiwane poziomy najlepszej efektywności.
- W trybie `real` metoda wybiera istniejące DMU najbliższe kolejnym poziomom `e_h`.
- W trybie `fictive` kandydat etapu `h` musi spełnić `best_efficiency >= e_h`.
- `milestone_gap` oznacza odchylenie od idealnego poziomu efektywności; im mniejsze, tym bardziej regularna ścieżka.
- Ścieżka kończąca się przy `best_efficiency = 1.0` osiąga pełną efektywność w sensie najlepszego dopuszczalnego układu wag.

## 5. Robustness Width Path

Score-based:

```powershell
python .\python\5_robustness_width_path_pipeline.py --input .\input\airports.csv --target KAT --target-width 0.15 --stages 3 --output-dir .\output --java-entry .\java --mode fictive --width-kind score --columns o1 --step-pct 50 --min-points-per-dim 3 --max-candidates 10 --points-per-stage 5 --max-paths 20
```

Rank-based:

```powershell
python .\python\5_robustness_width_path_pipeline.py --input .\input\airports.csv --target KAT --target-width 2 --stages 3 --output-dir .\output --java-entry .\java --mode fictive --width-kind rank --columns o1 --step-pct 50 --min-points-per-dim 3 --max-candidates 10 --points-per-stage 5 --max-paths 20
```

Warunki fictive z paperu:

- score-based: `W(z) = best_efficiency(z) - worst_efficiency(z)`
- rank-based: `W(z) = worst_rank(z) - best_rank(z)`
- na etapie `h`: `W(z_h) <= w_h`
- dodatkowo brak pogorszenia głównego wskaźnika postępu:
  - dla score: `best_efficiency(z_h) >= best_efficiency(z_{h-1})`
  - dla rank: `best_rank(z_h) <= best_rank(z_{h-1})`

Najważniejsze wyniki:

- `extreme_efficiencies.csv`
- `extreme_ranks.csv`
- `width_metrics.csv`
- `width_milestones.csv`
- `fictive_candidate_metrics.csv`
- `stage_candidates.csv`
- `paths.csv`

Interpretacja eksperymentu:

- `width_metrics.csv` pokazuje początkowe szerokości robustności dla istniejących DMU.
- `width_milestones.csv` pokazuje sekwencję `w_h`, czyli oczekiwane maksymalne szerokości na etapach.
- Dla `--width-kind score` szerokość to `best_efficiency - worst_efficiency`.
- Dla `--width-kind rank` szerokość to `worst_rank - best_rank`.
- Mniejszy width oznacza bardziej stabilną ocenę, czyli mniejszą wrażliwość na wybór wag.
- Sama redukcja width nie wystarcza; pipeline wymusza także brak pogorszenia głównego wskaźnika postępu.
- Brak ścieżek jest tutaj częsty i informacyjny: oznacza, że w danej siatce nie ma stanu, który jednocześnie zmniejsza width i nie pogarsza postępu.

## Jak Czytać Wyniki Eksperymentu

Najważniejsze pliki wspólne:

- `fictive_candidates.csv` - wszystkie wygenerowane sztuczne stany przed oceną robustDEA.
- `fictive_candidate_metrics.csv` - pełna ocena tych stanów przez Java/robustDEA.
- `stage_candidates.csv` - kandydaci, którzy przeszli filtr danego milestone'u.
- `paths.csv` - finalne ścieżki po uwzględnieniu monotoniczności input/output i limitu `--max-paths`.

Najważniejsze kolumny:

- `state_type` - `real` albo `fictive`.
- `best_efficiency` - najlepszy możliwy score; większy jest lepszy.
- `worst_efficiency` - najgorszy score; większy jest lepszy, ale interpretuj razem z width.
- `best_rank` - najlepsza możliwa ranga; mniejsza jest lepsza.
- `worst_rank` - najgorsza możliwa ranga; mniejsza jest lepsza.
- `score_width` - rozpiętość efektywności; mniejsza oznacza większą stabilność.
- `rank_width` - rozpiętość rankingu; mniejsza oznacza większą stabilność.
- `milestone_gap` - odchylenie od idealnego milestone'u; mniejsze jest lepsze.
- `effort_from_start` - uproszczona skala zmiany względem DMU startowego; mniejsze oznacza mniej agresywną modyfikację.
- `candidate_necessary_over_count` - liczba DMU, nad którymi kandydat ma necessary przewagę.
- `candidate_possible_over_count` - liczba DMU, nad którymi kandydat ma possible przewagę.

Praktyczna interpretacja:

- Jeśli `stage_candidates.csv` jest pusty, siatka nie znalazła punktów spełniających milestone.
- Jeśli `stage_candidates.csv` ma dane, ale `paths.csv` jest pusty, kandydaci istnieją, ale nie da się ich połączyć w monotoniczną ścieżkę zmian input/output.
- Jeśli `paths.csv` ma dużo wierszy, metoda znalazła wiele alternatywnych ścieżek; wtedy warto obniżyć `--points-per-stage` albo później użyć evaluatora ścieżek.
- Jeśli fictive states dają lepsze milestone'y niż real states, to oznacza, że w danych brakuje dobrych obserwowalnych benchmarków, ale przestrzeń osiągalna zawiera sensowne stany pośrednie.

## Uwaga O Koszcie

Tryb `real` jest szybki, bo wybiera spośród istniejących DMU. Tryb `fictive` jest znacznie cięższy, bo każdy wygenerowany stan jest oceniany przez modele robustDEA.

Najbezpieczniejsze pierwsze testy:

```powershell
python .\python\4_best_efficiency_path_pipeline.py --input .\input\airports.csv --target KAT --target-best-efficiency 1.0 --stages 2 --output-dir .\output\fictive_smoke --java-entry .\java --mode fictive --columns o1 --step-pct 50 --min-points-per-dim 3 --max-candidates 10 --points-per-stage 5 --max-paths 5
```

Potem stopniowo zwiększaj:

- liczbę kolumn w `--columns`
- liczbę punktów przez mniejszy `--step-pct`
- `--max-candidates`
- `--points-per-stage`
