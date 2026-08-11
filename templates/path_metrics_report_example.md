# Path Metrics Report

Source metrics CSV: `templates\path_metrics_output_example.csv`

## Candidate Paths

### Effort

| method | path_id | start_name | final_name | path_length | tc | msc | cdir | dr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| template | smooth_path | DMU_START | DMU_TARGET | 2 | 1 | 0.522727 | 1 | 1 |
| template | lumpy_path | DMU_START | DMU_TARGET | 2 | 1 | 0.912879 | 1 | 1 |

### Milestones

| method | path_id | start_name | final_name | path_length | md |
| --- | --- | --- | --- | --- | --- |
| template | smooth_path | DMU_START | DMU_TARGET | 2 | 0.025 |
| template | lumpy_path | DMU_START | DMU_TARGET | 2 | 0.1 |

### Balance

| method | path_id | start_name | final_name | path_length | bp | wbp | sbp | swbp | mcp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| template | smooth_path | DMU_START | DMU_TARGET | 2 | 0.00103306 | 0.00103306 | 0.00103306 | 0.00103306 | 0.545455 |
| template | lumpy_path | DMU_START | DMU_TARGET | 2 | 0.170622 | 0.170622 | 0.170622 | 0.170622 | 0.933333 |

### Progress

| method | path_id | start_name | final_name | path_length | pyv | pym |
| --- | --- | --- | --- | --- | --- | --- |
| template | smooth_path | DMU_START | DMU_TARGET | 2 | 0.000174481 | 0.35619 |
| template | lumpy_path | DMU_START | DMU_TARGET | 2 | 0.0124739 | 0.350539 |

### Robustness

| method | path_id | start_name | final_name | path_length | apw | fw | ww |
| --- | --- | --- | --- | --- | --- | --- | --- |
| template | smooth_path | DMU_START | DMU_TARGET | 2 | 0.14 | 0.08 | 0.14 |
| template | lumpy_path | DMU_START | DMU_TARGET | 2 | 0.153333 | 0.08 | 0.153333 |

### Reference continuity

| method | path_id | start_name | final_name | path_length | pc |
| --- | --- | --- | --- | --- | --- |
| template | smooth_path | DMU_START | DMU_TARGET | 2 | 0.666667 |
| template | lumpy_path | DMU_START | DMU_TARGET | 2 | 0.833333 |

### Operational profile

| method | path_id | start_name | final_name | path_length | opp |
| --- | --- | --- | --- | --- | --- |
| template | smooth_path | DMU_START | DMU_TARGET | 2 | 0.105211 |
| template | lumpy_path | DMU_START | DMU_TARGET | 2 | 0.0779257 |

### Realness

| method | path_id | start_name | final_name | path_length | rr |
| --- | --- | --- | --- | --- | --- |
| template | smooth_path | DMU_START | DMU_TARGET | 2 | 0.5 |
| template | lumpy_path | DMU_START | DMU_TARGET | 2 | 0.5 |

## Quick Selection Hints

- `tc` min: `smooth_path` = 1
- `msc` min: `smooth_path` = 0.522727
- `cdir` min: `smooth_path` = 1
- `dr` min: `smooth_path` = 1
- `md` min: `smooth_path` = 0.025
- `bp` min: `smooth_path` = 0.00103306
- `wbp` min: `smooth_path` = 0.00103306
- `sbp` min: `smooth_path` = 0.00103306
- `swbp` min: `smooth_path` = 0.00103306
- `mcp` min: `smooth_path` = 0.545455
- `pyv` min: `smooth_path` = 0.000174481
- `pym` max: `smooth_path` = 0.35619
- `apw` min: `smooth_path` = 0.14
- `fw` min: `smooth_path` = 0.08
- `ww` min: `smooth_path` = 0.14
- `pc` min: `smooth_path` = 0.666667
- `opp` min: `lumpy_path` = 0.0779257
- `rr` max: `smooth_path` = 0.5

## Metric Dictionary

| metric | group | direction | plain_description | paper_formula | notes |
| --- | --- | --- | --- | --- | --- |
| tc | Effort | lower_better | Total normalized path effort. Sums weighted normalized changes over all stages. | TC(pi) | Exact value depends on normalization ranges and factor weights; current template uses equal weights. |
| msc | Effort | lower_better | Worst single-stage normalized effort. Captures the hardest step. | MSC(pi) | Useful when abrupt one-stage changes are undesirable. |
| cdir | Effort | lower_better | Direct one-shot normalized effort from the start state to the final state. | Cdir | Used as denominator for DR. |
| dr | Effort | lower_better | Relative detour of the gradual path compared with direct movement. | DR(pi)=TC(pi)/Cdir | Values close to 1 mean little extra effort versus direct movement. |
| md | Milestones | lower_better | Average deviation from prescribed milestones. | MDg(pi) | Mirrors mean_milestone_gap. |
| bp | Balance | lower_better | Unweighted balance of changes over stages. Penalizes uneven distribution of each factor's total change. | BP(pi) | Zero means each active factor changes evenly over stages. |
| wbp | Balance | lower_better | Factor-weighted balance of changes over stages. | WBP(pi) | Current implementation uses equal factor weights unless customized later. |
| sbp | Balance | lower_better | Stage-weighted balance of changes over stages. | SBP(pi) | Current implementation uses equal stage weights unless customized later. |
| swbp | Balance | lower_better | Combined factor- and stage-weighted balance. | SWBP(pi) | With equal weights it is a directly comparable balance score. |
| mcp | Balance | lower_better | Largest single-stage share of a factor's total adjustment. | MCP(pi) | Closer to 1 means one stage absorbs most of at least one factor's change. |
| pyv | Progress | lower_better | Variance of progress gained per unit of stage effort. | PYV(pi) | Generic column uses the first available progress indicator; detailed variants are also available. |
| pym | Progress | higher_better | Worst progress yield among stages. | PYM(pi) | Higher means even the weakest stage gives reasonable progress. |
| apw | Robustness | lower_better | Average robustness interval width along the path. | APW(pi) | Generic column uses the first available width; score/rank variants are also available. |
| fw | Robustness | lower_better | Final robustness interval width. | FW(pi) | Measures robustness of the final recommendation. |
| ww | Robustness | lower_better | Weighted robustness width along the path. | WW(pi) | Current implementation uses equal stage weights unless customized later. |
| pc | Reference continuity | lower_better | Peer/reference set discontinuity between consecutive stages. | PC(pi) | Uses peer_refs/peer_set/reference_set if present; robust-reference fields are fallback variants. |
| opp | Operational profile | lower_better | L1 distance between each stage's input/output composition and the initial composition. | OPP(pi) | Small values mean the unit keeps a similar operating profile. |
| rr | Realness | higher_better | Share of intermediate/final stages that coincide with real observed units. | RR(pi) | 1 means all post-start stages are real; 0 means all are fictive. |

## Notes

- Lower-better metrics are cost-type criteria.
- Higher-better metrics are benefit-type criteria.
- Empty cells mean the required source columns were not available in the input `paths.csv`.
