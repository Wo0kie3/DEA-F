# EDU: DMU_03, five methods with a joint terminal goal

Open `five_method_report.html` in a browser after downloading this directory.
GitHub's file viewer displays HTML source, not the interactive report.
The report requires Internet access to load Plotly 3.3.1 from its CDN.
All chart data, metric selectors and calculation popups are included in the HTML.
The file is approximately 32.5 MB because it retains all five metric explorers.

## Experiment

- Data: `input/EDU.csv`; source and variable definitions: `input/EDU.metadata.json`.
- Target: DMU_03; modified inputs: i1, i2, i3; fixed output: o1 = 0.905.
- Common evaluated pool: 3,645 points (729 grid points and 2,916 jitter samples).
- Grid: 9 levels per input; 4 jitter samples per centre; radius: 0.6 grid steps;
  sampling seed: 20261019; Java evaluation batches: 500.
- At most 15 candidates per transition and 20,000 saved paths per method.
- Numeric methods: 3 stages; final thresholds: best rank 1, best efficiency 0.95,
  score width 0.15. Structural stage counts follow the NEC graph.
- All methods require a joint terminal comparison with DMU_05, DMU_16 and DMU_20
  using one nonnegative CCR weight vector for each terminal state.
- Costs and minimal-change fronts are calculated from the actual predecessor.

`experiment_params.csv` preserves the experiment's original parameter record,
including inherited source-run settings and local paths. The effective shared-pool
settings are `grid_levels`, `jitter_samples`, `jitter_fraction`, `sampling_seed`
and `evaluated_candidate_count`; inherited grid settings do not describe this pool.
Original dates and source commit identifiers refer to the recorded runs, not the
publication date of this example.

## Results and limitations

`selected_path_all_metrics.csv` contains the default paths' metric results.
`metric_explorer_selection.csv` records the path selected for each metric.
Coverage, exceptions, transition audits and validation checks are included.

Hasse and NEC-layer methods reached the 20,000-path cap. Their validation flags
`not_truncated_at_max_paths` are false. These results concern the saved, pruned
pool, not an exhaustive search of continuous space. Method 2 uses NEC ranking
layers; efficiency peeling is presented separately in the report.

This is a presentation snapshot, not the full computational archive: complete
candidate pools and all path CSVs remain in the local `output/` directory.
The HTML includes the displayed DEA input, matrices, fronts and diagrams.
`artifact_manifest.json` records file hashes and the CDN conversion; the conversion
preserves all scripts other than the embedded Plotly library.

## Regeneration From Local Results

With the original output directories available, run from the repository root:

```powershell
python python/generate_five_method_report.py `
  --root output/edu_five_joint_DMU03_20261019 `
  --input input/EDU.csv --target DMU_03 `
  --plotly-mode cdn `
  --output examples/edu_five_joint_DMU03_20261019/five_method_report.html
```

This regenerates reporting artifacts, not the experiment's search. See
`docs/EXPERIMENT_OUTPUT_AND_VISUALIZATION.md` for the experiment workflow.
