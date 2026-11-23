"""Register and run the paired one-action flop diagnostic on the frozen 480–504 policies."""
import argparse
from contextlib import closing
import fcntl
import gzip
import json
from pathlib import Path

from controlled_poker_refit import read, utc, verify_files
from fullhouse_runtime import BRIDGE, SOURCE as PUBLIC_SOURCE
from fullhouse_v3_common import ROOT, PYTHON, atomic_json, digest
from poker_multivalued_turn import RULES
from run_compact_refit_probe import check_card_range
from run_poker_candidate_confirmation import prior_card_ranges
from run_poker_preflop_confirmation import run_jobs

REL = Path('preflop-followup/480-v1/flop-action-diagnostic-20261123')
STUDY = ROOT/'artifacts'/REL
REPORT = ROOT/'reports'/REL
PARENT = STUDY.parent/'postflop-averaging-20261120'
CONFIG = ROOT/'configs/poker-flop-action-diagnostic-20261123.json'
SCRIPT = Path(__file__).resolve()


def register():
    cfg = read(CONFIG)
    registration = STUDY/'registration.json'
    if registration.exists():
        raise FileExistsError(registration)
    plan_file = PARENT/'population/evaluation-plan.json'
    frozen = {PARENT/'stage-ready.json':cfg['stage_sha256'],
              PARENT/'population/summary.json':cfg['parent_evaluation_summary_sha256'],
              plan_file:cfg['parent_evaluation_plan_sha256']}
    verify_files(frozen)
    parent = read(plan_file)
    verify_files(parent['frozen_files'])
    previous = prior_card_ranges(STUDY)
    ranges = {name: value['range'] for name, value in previous.items()}
    maximum = cfg['maximum_deals_scanned_per_seed']
    for seed in cfg['training_seeds']:
        start = cfg['first_deal_by_seed'][str(seed)]
        check_card_range(start, maximum, ranges)
        ranges[f'current-{seed}'] = (start, start+maximum)
    public = read(ROOT/'configs/fullhouse-model.json')
    frozen.update({PUBLIC_SOURCE/'data'/name:sha for name,sha in public['files'].items()})
    frozen.update({p:digest(p) for p in [BRIDGE, RULES, ROOT/'configs/fullhouse-model.json',
                 ROOT/'scripts/poker_reference_opponents.py']})
    verify_files(frozen)
    registration_value = dict(created_utc=utc(), config_sha256=digest(CONFIG),
                    historical_ranges_checked=len(previous),
                    previous_ranges=[list(value['range']) for value in previous.values()],
                    parent_bound_files=len(parent['frozen_files']),
                    frozen_files={str(path):value for path,value in frozen.items()},
                    first_deal_by_seed=cfg['first_deal_by_seed'], reserved_length=maximum)
    for seed,start in cfg['first_deal_by_seed'].items():
        folder = STUDY/'reservations'/seed
        folder.mkdir(parents=True,exist_ok=True)
        atomic_json(folder/'config.json',dict(evaluation_deal_seed=start,evaluation_deals=maximum))
    atomic_json(registration,registration_value)
    REPORT.mkdir(parents=True,exist_ok=True)
    atomic_json(REPORT/'registration.json',registration_value)


def bound():
    reg = read(STUDY/'registration.json')
    verify_files({CONFIG:reg['config_sha256'], **reg['frozen_files']})
    verify_files(read(PARENT/'population/evaluation-plan.json')['frozen_files'])
    return read(CONFIG)


def rollout_worker(seed, cfg):
    from poker_flop_action_values import rollout_state
    from poker_preflop_specialist import evaluator_policy
    from poker_multivalued_turn import Rules
    from fullhouse_runtime import Bridge, Fullhouse
    from poker_reference_opponents import ReferenceOpponent

    folder = STUDY/f'seed-{seed}'
    sample = folder/'states.jsonl'
    bound_states = read(STUDY/'rollout-inputs.json')['states']
    verify_files({sample:bound_states[str(seed)]})
    plan = read(PARENT/'population/evaluation-plan.json')
    ref = plan['policies']['specialist480'][str(seed)]
    parent = evaluator_policy(ref['path'],plan['source'])
    bots = {'self480':parent,'public':Fullhouse(),'tight-value':ReferenceOpponent('tight-value')}
    count = branches = recorded_traces = 0
    values = folder/'values.jsonl.gz'
    traces = folder/'traces.jsonl.gz'
    with sample.open() as states, closing(Bridge()) as bridge, closing(Rules()) as rules, \
            gzip.open(values,'xt',compresslevel=2) as value_out, gzip.open(traces,'xt',compresslevel=2) as trace_out:
        for line in states:
            state = json.loads(line)
            if state['seed'] != seed:
                raise ValueError('Sample assigned to the wrong seed')
            rows, first_repeats = rollout_state(state, bridge, rules, parent, bots, repeats=cfg['paired_repetitions'])
            if len(rows)!=len(bots) or {row['lineup'] for row in rows}!=set(bots):
                raise ValueError('Incomplete opponent comparisons')
            for row in rows:
                value_out.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n')
            for trace in first_repeats:
                trace_out.write(json.dumps(trace,separators=(',',':'),allow_nan=False)+'\n')
            count += 1
            recorded_traces += len(first_repeats)
            branches += sum(sum(len(v) for v in row['action_payoffs_chips'] if v is not None) for row in rows)
            if count%64 == 0:
                atomic_json(folder/'rollout-progress.json',dict(utc=utc(),states=count,terminal_branches=branches))
    if count!=cfg['eligible_original_deals_per_seed'] or branches!=cfg['paired_repetitions']*recorded_traces:
        raise ValueError('Rollout counts do not reconcile')
    atomic_json(folder/'rollout-complete.json',dict(utc=utc(),seed=seed,states=count,
                terminal_branches=branches,trace_count=recorded_traces,source_sha256=digest(sample),
                values_sha256=digest(values), traces_sha256=digest(traces)))


def phase(name):
    cfg = bound()
    seeds = cfg['training_seeds']
    sources = {'sample':['sample_poker_flop_diagnostic.py'],
               'rollout':['poker_flop_action_values.py']}[name]
    code = {str(path):digest(path) for path in [SCRIPT,*(SCRIPT.parent/file for file in sources)]}
    inputs = {'utc':utc(),'code':code,'registration_sha256':digest(STUDY/'registration.json')}
    if name=='rollout':
        verify_files(read(STUDY/'sample-inputs.json')['code'])
        if read(STUDY/'sample-status.json')['state']!='complete':
            raise ValueError('Sampling has not completed')
        inputs['states'] = {str(s):digest(STUDY/f'seed-{s}/states.jsonl') for s in seeds}
    inputs_file = STUDY/f'{name}-inputs.json'
    if inputs_file.exists():
        raise FileExistsError(inputs_file)
    atomic_json(inputs_file,inputs)
    commands = {f'seed-{s}':[str(PYTHON),str(SCRIPT),'_'+name,str(s)] for s in seeds}
    run_jobs(STUDY,commands,len(seeds),name)
    verify_files(code)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['register','sample','rollout','_sample','_rollout'])
    parser.add_argument('seed',type=int,nargs='?')
    args=parser.parse_args()
    STUDY.mkdir(parents=True,exist_ok=True)
    if args.phase.startswith('_'):
        cfg=bound()
        if args.seed not in cfg['training_seeds']:
            parser.error('Unexpected seed')
        verify_files(read(STUDY/(args.phase[1:]+'-inputs.json'))['code'])
        if args.phase=='_sample':
            from sample_poker_flop_diagnostic import sample_seed
            sample_seed(args.seed,cfg['first_deal_by_seed'][str(args.seed)],cfg['eligible_original_deals_per_seed'],
                        cfg['maximum_deals_scanned_per_seed'],STUDY)
        else:
            rollout_worker(args.seed,cfg)
    else:
        with (STUDY/'workflow.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if args.phase=='register':
                register()
            else:
                phase(args.phase)
