"""Run the original five methods on one evaluated Hasse/POS pool, sequentially."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import pandas as pd

from generate_five_method_robot_report import _latest_run, _selected_path
from generate_hasse_pos_report import build_report


def main(pool, output_root=None, final_goal='method-specific', max_paths=None):
    pool = pool.resolve()
    root = output_root.resolve() if output_root else pool/('five_methods_joint' if final_goal == 'joint' else 'five_methods')
    if root.exists():
        raise ValueError('Use a fresh five_methods directory')
    root.mkdir(parents=True)
    shutil.copytree(pool/'dea_input_snapshot', root/'dea_input_snapshot')
    params = dict(pd.read_csv(pool/'experiment_params.csv', keep_default_na=False).itertuples(index=False, name=None))
    params.update(root=str(root), shared_pool=str(pool), final_goal=final_goal, candidate_pool_note='identical evaluated pool; no extra refinement',
                  refine_iterations=0, local_samples=0, global_samples=0)
    if max_paths is not None:
        params['max_paths'] = max_paths
    pd.DataFrame(params.items(), columns=['parameter','value']).to_csv(root/'experiment_params.csv', index=False)
    repo = Path(__file__).resolve().parents[1]
    records=[]
    def run(label, script, args, allow_review=False):
        command=[sys.executable, str(repo/'python'/script), *map(str,args)]
        print('Starting '+label, flush=True)
        start=time.perf_counter()
        with (root/(label+'.log')).open('w',encoding='utf-8') as log:
            result=subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        records.append(dict(step=label, elapsed_seconds=time.perf_counter()-start, returncode=result.returncode, command_json=json.dumps(command)))
        pd.DataFrame(records).to_csv(root/'execution_status.csv',index=False)
        if result.returncode and not allow_review:
            raise RuntimeError(f'{label} failed: {root/(label+".log")}')
    common=['--input',root/'dea_input_snapshot/decision_data.csv','--target',params['target'],
        '--output-dir',root,'--java-entry',repo/'java','--mode','fictive','--columns',params['dimensions'],
        '--shared-pool',pool,'--points-per-stage',params['points_per_transition'], '--max-paths',params['max_paths'],
        '--factor-weights',params['factor_weights'],'--final-goal',final_goal]
    for key in ('max_stage_effort', 'max_factor_changes'):
        if params.get(key, 'none') != 'none':
            common += ['--'+key.replace('_','-'), params[key]]
    methods=[('hasse_path','1_hasse_path_pipeline.py',[]),
        ('front_path','2_necessary_layer_path_pipeline.py',[]),
        ('best_rank_path','3_best_rank_path_pipeline.py',['--stages',3,'--target-best-rank',params['target_best_rank']]),
        ('best_efficiency_path','4_best_efficiency_path_pipeline.py',['--stages',3,'--target-best-efficiency',params['target_best_efficiency']]),
        ('robustness_width_path','5_robustness_width_path_pipeline.py',['--stages',3,'--target-width',params['target_width'],'--width-kind',params['width_kind']])]
    for label,script,extra in methods:
        run(label,script,common+extra)
    run('completion','complete_experiment_metrics.py',['--root',root,'--java-entry',repo/'java'])
    run('validation','validate_controlled_experiment.py',['--root',root,'--fixed-columns',params['fixed_columns'],'--max-paths',params['max_paths']], allow_review=True)
    checks = pd.read_csv(root/'validation_checks.csv')
    errors = checks.loc[~checks.passed & ~checks.check.eq('not_truncated_at_max_paths')]
    if not errors.empty:
        raise ValueError('Numerical validation failed: '+errors.to_string(index=False))
    if final_goal == 'joint':
        from generate_five_method_report import build_report as build_five_report
        build_five_report(root,root/'dea_input_snapshot/decision_data.csv',params['target'],root/'five_method_report.html')
        print(root/'five_method_report.html')
        return
    selected={}
    for variant,color in [('pairwise','#d97706'),('joint','#2563b6')]:
        metric,path,*_=_selected_path(pool/variant)
        selected[variant]=(metric,path,color)
    build_report(pool,pd.read_csv(pool/'dea_input_snapshot/decision_data.csv'),pd.read_csv(pool/'evaluated_candidates.csv'),
        selected,pd.read_csv(pool/'variant_summary.csv'),params,pd.read_csv(pool/'top_targets.csv').name.tolist(),five_method_root=root)
    print(pool/'hasse_pos_comparison.html')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pool',type=Path,required=True)
    parser.add_argument('--output-root',type=Path)
    parser.add_argument('--final-goal',choices=['method-specific','joint'],default='method-specific')
    parser.add_argument('--max-paths',type=int,help='Override for a smoke test or controlled run')
    args=parser.parse_args()
    main(args.pool,args.output_root,args.final_goal,args.max_paths)
