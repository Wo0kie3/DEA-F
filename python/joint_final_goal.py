"""A common-weight terminal constraint, applied before transition Pareto selection."""
import json
from pathlib import Path

import pandas as pd
from pandas.api.types import is_bool_dtype


def filter_joint_final(pool, targets):
    if 'joint_top' not in pool or not is_bool_dtype(pool.joint_top.dtype) or pool.joint_top.isna().any():
        raise ValueError('A fully evaluated boolean joint_top is required; pairwise POS is insufficient')
    result = pool.loc[pool.joint_top].copy()
    result['milestone_joint_required'] = True
    result['milestone_joint_targets'] = '|'.join(targets)
    return result


def initialize_joint_goal(args, run):
    if args.final_goal != 'joint':
        return None
    if not args.shared_pool or args.mode != 'fictive':
        raise ValueError('--final-goal joint currently requires --shared-pool and --mode fictive')
    from hasse_pos_goals import build_pos_routes
    root = Path(args.shared_pool)
    matrix = pd.read_csv(root/'dea_input_snapshot/necessary_matrix.csv',index_col=0)
    _, targets, _, _ = build_pos_routes(matrix,args.target)
    if targets != pd.read_csv(root/'top_targets.csv').name.tolist():
        raise ValueError('Shared common-weight target set differs from global NEC maxima')
    for filename in ('top_targets.csv','possible_goals.csv','joint_weight_audit.csv'):
        if not (root/filename).is_file():
            raise ValueError('Missing common-weight evidence: '+filename)
    (run/'final_goal.json').write_text(json.dumps(dict(final_goal='joint',targets=targets,
        evidence_root=str(root.resolve()), semantics='one common nonnegative CCR weight vector at the final state; not one vector for the entire path'),indent=2),encoding='utf-8')
    return targets


def write_joint_hasse(args, run, start, candidates, inputs, outputs, columns, ranges, weights, max_factor_changes):
    from hasse_pos_goals import build_pos_routes, enumerate_pos_variant
    from path_selection import write_publication_selection
    matrix = pd.read_csv(Path(args.shared_pool)/'dea_input_snapshot/necessary_matrix.csv',index_col=0)
    routes, targets, _, _ = build_pos_routes(matrix,args.target)
    paths, metrics, pools, transitions, capped = enumerate_pos_variant(start,candidates,routes,targets,'joint',
        inputs,outputs,columns,ranges,weights,points_per_transition=args.points_per_stage,max_paths=args.max_paths,
        max_stage_effort=args.max_stage_effort,max_factor_changes=max_factor_changes)
    final = pools.milestone_goal_kind.eq('joint_top')
    pools['milestone_joint_required'] = final
    pools['milestone_joint_targets'] = '|'.join(targets)
    for filename,frame in [('paths.csv',paths),('path_metrics.csv',metrics),('stage_candidates.csv',pools),('transition_candidates.csv',transitions)]:
        frame.to_csv(run/filename,index=False)
    (run/'graph_routes.json').write_text(json.dumps(dict(routes=routes,top=targets,max_paths_reached=capped),indent=2),encoding='utf-8')
    write_publication_selection(metrics,run,method_name='hasse_path')
