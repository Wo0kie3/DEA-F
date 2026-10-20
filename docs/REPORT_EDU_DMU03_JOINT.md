# Raport referencyjny EDU-DMU03-JOINT

Stały identyfikator raportu: **EDU-DMU03-JOINT**.
Ta karta służy do odwoływania się do konkretnego eksperymentu w kolejnych wątkach.
Zmiana metod lub ponowne obliczenie eksperymentu powinny otrzymać osobny identyfikator.

## Gdzie Znaleźć Raport

- [Raport HTML w repozytorium](../examples/edu_five_joint_DMU03_20261019/five_method_report.html).
- [Opis pakietu i parametry](../examples/edu_five_joint_DMU03_20261019/README.md).
- [Parametry CSV](../examples/edu_five_joint_DMU03_20261019/experiment_params.csv).
- [Metryki domyślnie wybranych ścieżek](../examples/edu_five_joint_DMU03_20261019/selected_path_all_metrics.csv).
- [Wybory ścieżek według metryk](../examples/edu_five_joint_DMU03_20261019/metric_explorer_selection.csv).
- [Niezmienny snapshot na GitHubie, commit 292da3b](https://github.com/Wo0kie3/DEA-F/tree/292da3bded3f4600d34298184dfb1865c4f6ca09/examples/edu_five_joint_DMU03_20261019).

Pełna lokalna wersja offline:
`output/edu_five_joint_DMU03_20261019/five_method_report.html`.
Pełne dane obliczeniowe pozostają w tym katalogu oraz we wspólnej puli
`output/edu_seven_methods_DMU03_20261019/`.
Wersja HTML w `examples/` zawiera dane wykresów, ale wymaga Internetu do pobrania Plotly.
GitHub wyświetla kod HTML; interaktywny raport otwiera się po pobraniu w przeglądarce.

## Co Badamy

- Dane: 20 lekarzy pediatrycznego oddziału ratunkowego CHEO; grupa pacjentów:
  ból brzucha i zaparcia, 2011 r. Źródło: [opis EDU](EDU_DATA_PROVENANCE.md).
- Start: DMU_03; zmieniane i1, i2, i3; stałe o1 = 0.905.
- Pięć metod: ścieżka Hassego, warstwy rankingu NEC, najlepsza ranga,
  najlepsza efektywność, szerokość oceny.
- Wspólny cel końcowy: jeden nieujemny wektor wag CCR dla jednoczesnego porównania
  końca ścieżki z DMU_05, DMU_16 i DMU_20. Wagi mogą różnić się między stanami.
- Hasse wymaga POS nad wskazaną DMU na etapach pośrednich. Metoda warstw NEC
  wymaga necessary nad pełną klasą danej warstwy. Metody liczbowe zachowują swoje progi.
- Pula: 3645 punktów; 9 poziomów siatki, 4 próbki szumu na centrum,
  promień 0.6 kroku siatki, seed 20261019.
- Koszty i fronty minimalnych zmian są liczone względem rzeczywistego poprzednika.

## Co Zawiera Raport

Dane wejściowe, scores i rangi DEA, macierze NEC/POS, fronty efektywności z peelingu,
ranking NEC i diagram Hassego, pięć paneli ścieżek z wyborem spośród 20 pozycji
katalogu metryk, rachunki metryk, etapy, audyty i wagi potwierdzające cel końcowy.

Wnioski dotyczą ocenionej puli po selekcji frontów i limitach. Dwie metody strukturalne
osiągnęły limit 20000 zapisanych ścieżek. Nie jest to pełna enumeracja przestrzeni
ciągłej. Warstwy rankingu NEC, fronty efektywności i fronty minimalnych zmian są
odrębnymi pojęciami. PC wykorzystuje sumę radialnie optymalnych zbiorów peerów CCR.

## Wiadomość Do Innego Wątku

> Pracujemy w repozytorium DEA-F. Odwołuję się do raportu EDU-DMU03-JOINT.
> Przeczytaj docs/REPORT_EDU_DMU03_JOINT.md, a następnie wskazany raport i parametry.
> To eksperyment pięciu metod dla DMU_03 ze wspólnym końcowym wektorem wag CCR.
> Zachowaj rozróżnienie celów metod i ograniczeń ocenionej puli.
