"""Load one evaluated pool for controlled comparisons, or retain legacy generation."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from java_runner import export_candidate_robust_metrics_with_java
from path_pipeline_common import generate_attainable_fictive_candidates, get_io_columns, path_for_java


def load_shared_pool(root, reference, target, columns):
    root = Path(root)
    pd.testing.assert_frame_equal(reference, pd.read_csv(root/'dea_input_snapshot/decision_data.csv'), check_dtype=False)
    params = dict(pd.read_csv(root/'experiment_params.csv', keep_default_na=False).itertuples(index=False, name=None))
    if params['target'] != target['name'] or params['dimensions'].split(',') != list(columns):
        raise ValueError('Shared pool target/dimensions mismatch')
    path = root/'evaluated_candidates.csv'
    frame = pd.read_csv(path)
    inputs, outputs = get_io_columns(reference)
    io = inputs+outputs
    if frame.empty or frame.name.duplicated().any() or not np.isfinite(frame[io].to_numpy()).all():
        raise ValueError('Invalid shared candidate identities or coordinates')
    if (frame[io] <= 0).any().any():
        raise ValueError('Shared candidates must have positive coordinates')
    if (frame[inputs] > target[inputs].astype(float)+1e-9).any().any() or (frame[outputs] < target[outputs].astype(float)-1e-9).any().any():
        raise ValueError('Unattainable shared candidates')
    for col in set(io)-set(columns):
        if not np.allclose(frame[col], target[col], atol=1e-12, rtol=0):
            raise ValueError('Shared pool modifies fixed factor')
    return frame, dict(file=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), count=len(frame))


def prepare_fictive_metrics(args, reference, target, columns, run):
    if args.shared_pool:
        if getattr(args, 'refine_fictive_candidates', False) or getattr(args, 'global_search_samples', 0):
            raise ValueError('Shared-pool comparison must not add method-specific samples')
        metrics, manifest = load_shared_pool(args.shared_pool, reference, target, columns)
        (run/'shared_pool_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        inputs, outputs = get_io_columns(reference)
        metrics[['name', 'state_type', *inputs, *outputs]].to_csv(run/'fictive_candidates.csv', index=False)
        metrics.to_csv(run/'fictive_candidate_metrics.csv', index=False)
        return metrics
    candidates = generate_attainable_fictive_candidates(df=reference, target_row=target,
        columns_to_modify=columns, pct_above=args.pct_above, step_pct=args.step_pct,
        step_abs=args.step_abs, min_points_per_dim=args.min_points_per_dim,
        max_candidates=args.max_candidates, name_prefix=args.target)
    candidates_csv = run/'fictive_candidates.csv'
    candidates.to_csv(candidates_csv, index=False)
    metrics_csv = run/'fictive_candidate_metrics.csv'
    export_candidate_robust_metrics_with_java(reference_csv=path_for_java(args.input, args.java_entry),
        candidates_csv=path_for_java(str(candidates_csv), args.java_entry),
        output_csv=path_for_java(str(metrics_csv), args.java_entry), java_entry=args.java_entry,
        main_class=args.candidate_metrics_main_class, maven_executable=args.maven_executable)
    return pd.read_csv(metrics_csv)
