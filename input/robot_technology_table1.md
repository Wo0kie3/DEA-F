# Robot technology dataset - Table 1

Plik `robot_technology_table1.csv` odtwarza 27 robotów z Table 1 artykułu:

R. C. Baker, S. Talluri, *A closer look at the use of data envelopment analysis for technology selection*, Computers & Industrial Engineering 32(1), 101-108, 1997.

## Mapowanie kolumn

| CSV | Table 1 | Rola DEA | Jednostka | Kierunek |
|---|---|---|---|---|
| `name` | Robot No. | identyfikator DMU | - | - |
| `i1` | Cost | input | $10,000 | minimalizacja |
| `i2` | Repeatability | input | mm | minimalizacja |
| `o1` | Load capacity | output | kg | maksymalizacja |
| `o2` | Velocity | output | m/s | maksymalizacja |
| `reported_efficiency` | Eff. | kontrola względem publikacji | - | nie jest kolumną DEA |

`reported_efficiency` zawiera wartości zaokrąglone do dwóch miejsc w artykule. Obliczenia DEA-F używają wyłącznie kolumn `i1`, `i2`, `o1` i `o2`.
