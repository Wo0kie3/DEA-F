# Raport Metryk Ścieżek DEA

Źródłowy plik metryk: `templates\path_metrics_output_example.csv`

## Zakres

- Liczba ścieżek w raporcie: **2**
- Liczba metod z niepustymi ścieżkami: **1**

## Ścieżki Kandydackie Według Grup Metryk

### Wysiłek

**Opis metryk w tej grupie**

| metric | direction | plain_description | paper_formula | notes |
| --- | --- | --- | --- | --- |
| tc | lower_better | Całkowity znormalizowany wysiłek ścieżki, czyli suma ważonych zmian między kolejnymi punktami wszystkich etapów. | TC(pi) | Koszt obejmuje czynniki modyfikowane; domyślnie ich wagi są równe, a zakresy pochodzą z danych referencyjnych. |
| msc | lower_better | Największy wysiłek pojedynczego etapu; pokazuje najtrudniejszy krok. | MSC(pi) | Przydatne, gdy chcemy unikać gwałtownych zmian w jednym etapie. |
| cdir | lower_better | Najmniejszy znormalizowany wysiłek bezpośredni do celu znaleziony w ocenionej puli końcowej. | Cdir | Mianownik DR; cdir_scope podaje zakres skończonej puli, więc nie jest to analityczne minimum ciągłe. |
| cdir_endpoint | lower_better | Znormalizowany wysiłek bezpośredni od startu do końca konkretnej ścieżki. | - | Wartość opisowa różna od wspólnego Cdir. |
| dr | lower_better | Relatywny objazd ścieżki etapowej względem ruchu bezpośredniego. | DR(pi)=TC(pi)/Cdir | Wartość bliska 1 oznacza, że etapowanie nie zwiększa istotnie całkowitego wysiłku. |

**Wartości dla ścieżek**

| method | path_id | start_name | final_name | path_length | tc | msc | cdir | cdir_endpoint | dr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| template | smooth_path | DMU_START | DMU_TARGET | 2 | 1 | 0.522727 | 1 | 1 | 1 |
| template | lumpy_path | DMU_START | DMU_TARGET | 2 | 1 | 0.912879 | 1 | 1 | 1 |

### Kamienie milowe

**Opis metryk w tej grupie**

| metric | direction | plain_description | paper_formula | notes |
| --- | --- | --- | --- | --- |
| md | lower_better | Średnie odchylenie od zaplanowanych milestone'ów. | MDg(pi) | Odpowiada kolumnie mean_milestone_gap. |

**Wartości dla ścieżek**

| method | path_id | start_name | final_name | path_length | md |
| --- | --- | --- | --- | --- | --- |
| template | smooth_path | DMU_START | DMU_TARGET | 2 | 0.025 |
| template | lumpy_path | DMU_START | DMU_TARGET | 2 | 0.1 |

### Równomierność

**Opis metryk w tej grupie**

| metric | direction | plain_description | paper_formula | notes |
| --- | --- | --- | --- | --- |
| bp | lower_better | Nieważona równomierność zmian w etapach; karze nierówny rozkład całkowitej zmiany danego czynnika. | BP(pi) | Zero oznacza, że każdy aktywny czynnik zmienia się równo w etapach. |
| wbp | lower_better | Równomierność zmian z wagami czynników. | WBP(pi) | Wagi są równe domyślnie albo jawne przez --factor-weights. |
| sbp | lower_better | Równomierność zmian z wagami etapów. | SBP(pi) | Wagi są równe domyślnie albo jawne przez --stage-weights. |
| swbp | lower_better | Równomierność zmian z wagami czynników i etapów jednocześnie. | SWBP(pi) | Przy równych wagach jest porównywalna z BP. |
| mcp | lower_better | Największy udział jednego etapu w całkowitej zmianie dowolnego czynnika. | MCP(pi) | Wartość bliska 1 oznacza, że jeden etap przejmuje prawie całą zmianę któregoś czynnika. |

**Wartości dla ścieżek**

| method | path_id | start_name | final_name | path_length | bp | wbp | sbp | swbp | mcp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| template | smooth_path | DMU_START | DMU_TARGET | 2 | 0.00103306 | 0.00103306 | 0.00103306 | 0.00103306 | 0.545455 |
| template | lumpy_path | DMU_START | DMU_TARGET | 2 | 0.170622 | 0.170622 | 0.170622 | 0.170622 | 0.933333 |

### Postęp

**Opis metryk w tej grupie**

| metric | direction | plain_description | paper_formula | notes |
| --- | --- | --- | --- | --- |
| pyv | lower_better | Zmienność postępu uzyskanego z jednostki wysiłku w kolejnych etapach. | PYV(pi) | Kolumna ogólna używa wskaźnika postępu przypisanego do metody; warianty szczegółowe są w CSV. |
| pym | higher_better | Najgorszy etapowy postęp na jednostkę wysiłku. | PYM(pi) | Wyższa wartość oznacza, że nawet najsłabszy etap daje sensowny postęp. |

**Wartości dla ścieżek**

| method | path_id | start_name | final_name | path_length | pyv | pym |
| --- | --- | --- | --- | --- | --- | --- |
| template | smooth_path | DMU_START | DMU_TARGET | 2 | 0.000174481 | 0.35619 |
| template | lumpy_path | DMU_START | DMU_TARGET | 2 | 0.0124739 | 0.350539 |

### Robustność

**Opis metryk w tej grupie**

| metric | direction | plain_description | paper_formula | notes |
| --- | --- | --- | --- | --- |
| apw | lower_better | Średnia szerokość przedziału robustności na ścieżce. | APW(pi) | Kolumna ogólna używa score_width albo rank_width zgodnie z metodą. |
| fw | lower_better | Końcowa szerokość przedziału robustności. | FW(pi) | Mierzy stabilność końcowej rekomendacji. |
| ww | lower_better | Ważona szerokość robustności na ścieżce. | WW(pi) | Wagi są równe domyślnie albo jawne przez --width-state-weights. |

**Wartości dla ścieżek**

| method | path_id | start_name | final_name | path_length | apw | fw | ww |
| --- | --- | --- | --- | --- | --- | --- | --- |
| template | smooth_path | DMU_START | DMU_TARGET | 2 | 0.14 | 0.08 | 0.14 |
| template | lumpy_path | DMU_START | DMU_TARGET | 2 | 0.153333 | 0.08 | 0.153333 |

### Ciągłość benchmarków

**Opis metryk w tej grupie**

| metric | direction | plain_description | paper_formula | notes |
| --- | --- | --- | --- | --- |
| pc | lower_better | Nieciągłość prawdziwych zbiorów peer/reference między kolejnymi etapami. | PC(pi) | Pozostaje puste, jeżeli adapter nie dostarcza peer sets. |

**Wartości dla ścieżek**

| method | path_id | start_name | final_name | path_length | pc |
| --- | --- | --- | --- | --- | --- |
| template | smooth_path | DMU_START | DMU_TARGET | 2 | 0.666667 |
| template | lumpy_path | DMU_START | DMU_TARGET | 2 | 0.833333 |

### Profil operacyjny

**Opis metryk w tej grupie**

| metric | direction | plain_description | paper_formula | notes |
| --- | --- | --- | --- | --- |
| opp | lower_better | Odległość L1 między strukturą input/output w etapach a strukturą początkową. | OPP(pi) | Mała wartość oznacza, że jednostka zachowuje podobny profil działalności. |

**Wartości dla ścieżek**

| method | path_id | start_name | final_name | path_length | opp |
| --- | --- | --- | --- | --- | --- |
| template | smooth_path | DMU_START | DMU_TARGET | 2 | 0.105211 |
| template | lumpy_path | DMU_START | DMU_TARGET | 2 | 0.0779257 |

### Rzeczywistość ścieżki

**Opis metryk w tej grupie**

| metric | direction | plain_description | paper_formula | notes |
| --- | --- | --- | --- | --- |
| rr | higher_better | Udział etapów po starcie, które odpowiadają realnym obserwowanym jednostkom. | RR(pi) | 1 oznacza, że wszystkie etapy po starcie są realne; 0 oznacza same punkty fictive. |

**Wartości dla ścieżek**

| method | path_id | start_name | final_name | path_length | rr |
| --- | --- | --- | --- | --- | --- |
| template | smooth_path | DMU_START | DMU_TARGET | 2 | 0.5 |
| template | lumpy_path | DMU_START | DMU_TARGET | 2 | 0.5 |

## Szybkie Wskazówki Selekcji

- `tc` min: `template` / `smooth_path` = 1
- `msc` min: `template` / `smooth_path` = 0.522727
- `cdir` min: `template` / `smooth_path` = 1
- `cdir_endpoint` min: `template` / `smooth_path` = 1
- `dr` min: `template` / `smooth_path` = 1
- `md` min: `template` / `smooth_path` = 0.025
- `bp` min: `template` / `smooth_path` = 0.00103306
- `wbp` min: `template` / `smooth_path` = 0.00103306
- `sbp` min: `template` / `smooth_path` = 0.00103306
- `swbp` min: `template` / `smooth_path` = 0.00103306
- `mcp` min: `template` / `smooth_path` = 0.545455
- `pyv` min: `template` / `smooth_path` = 0.000174481
- `pym` max: `template` / `smooth_path` = 0.35619
- `apw` min: `template` / `smooth_path` = 0.14
- `fw` min: `template` / `smooth_path` = 0.08
- `ww` min: `template` / `smooth_path` = 0.14
- `pc` min: `template` / `smooth_path` = 0.666667
- `opp` min: `template` / `lumpy_path` = 0.0779257
- `rr` max: `template` / `smooth_path` = 0.5

## Uwagi

- Metryki `lower_better` są kryteriami kosztowymi.
- Metryki `higher_better` są kryteriami korzyści.
- Opisy metryk są umieszczone bezpośrednio przed tabelą wartości dla danej grupy.
- Jeśli raport używa próbki, tabele ścieżek pokazują tylko wylosowane rekordy; podsumowania metod liczone są z pełnego CSV.
- Puste komórki oznaczają, że wymagane kolumny źródłowe nie były dostępne w wejściowym `paths.csv`.
