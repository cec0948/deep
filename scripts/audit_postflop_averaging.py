"""Independently verify the four-snapshot learning window and its matched-card evaluation."""
import argparse
import ast
from collections import Counter
from contextlib import ExitStack, closing
from functools import lru_cache
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
from statistics import NormalDist

import numpy as np

from audit_poker_population import check, close_to, expected_identities, interval, matched_role
from controlled_poker_refit import read, utc, verify_files
from evaluate_average_baseline import hand_seed, semantic_seed
from fullhouse_runtime import Bridge, Fullhouse
from fullhouse_v3_common import ROOT, atomic_json, digest
from poker_preflop_specialist import evaluator_policy
from poker_reference_opponents import ReferenceOpponent
from rich_poker_runtime import RichPolicy
from run_poker_candidate_confirmation import prior_card_ranges
from snapshot_poker_preflop_online import verify_export
from train_poker_average_continuation import load_checkpoint


BASE = ROOT/'artifacts/preflop-followup/480-v1'
STUDY = BASE/'postflop-averaging-20261120'
OLD = BASE/'fixed-preflop-504'
PARENT, MIX, END, DOSE = 'specialist480', 'mixture504', 'slow504', 'steps256'
SEEDS = (342, 343, 344)
SLOTS = (486, 492, 498, 504)
RICH = ROOT/'artifacts/rich-selfplay/20261113T075620400993Z/stage-ready.json'
FIELD_CARDS = {field: (2038500000, 2000) for field in
               ('noregrets', 'rich96', 'calling-station', 'tight-value', 'pressure', 'mixed')}
FIELD_CARDS.update(public=(2038400000, 20000), varied=(2038600000, 2000))
SAMPLE_STRIDE = 311


def net_fingerprint(export, mapping):
    keys = {'main_scales': 'main_scales', 'opening_scale': 'opening_scale'}
    for head, prefix in (('main', ''), ('opening', 'opening_')):
        keys.update({f'{head}.{key}': prefix + value for key, value in mapping.items()})
    h = hashlib.sha256()
    with np.load(export, allow_pickle=False) as arrays:
        for key, stored in keys.items():
            data = arrays[stored]
            check(np.isfinite(data).all(), f'Nonfinite weight in {export}')
            h.update(key.encode())
            h.update(data.tobytes())
    return h.hexdigest()


def checked_ref(ref):
    verify_files({ref['path']: ref['sha256']})
    return Path(ref['path'])


def seed_training(seed, plan, policies, mapping):
    key = str(seed)
    folder = STUDY/END/f'seed-{seed}'
    start, ready, ptr = (read(folder/name) for name in ('start-verification.json', 'ready.json', 'current.json'))
    verify_files({folder/'diagnostics.jsonl': ready['diagnostics_sha256'], folder/'start-verification.json': ready['start_sha256']})
    entry_ref, end_ref, mixture_ref = (policies[name][key] for name in (PARENT, END, MIX))
    source_policy, end_policy, mix = (read(checked_ref(ref)) for ref in (entry_ref, end_ref, mixture_ref))
    parent_ck = Path(plan['campaign'])/f'seed-{seed}'
    source, checkpoint = load_checkpoint(parent_ck)
    trained, end = load_checkpoint(folder)
    check(source['config'] == read(parent_ck.parent/'config.json') and source['iteration'] == 480 and trained['iteration'] == 504 and source['seed'] == trained['seed'] == seed
          and trained['config']['learning_rate'] == .00025 and source['config']['learning_rate'] == .001
          and ptr['iteration'] == 504 and ready['files'] == ptr['files']
          and start['checkpoint_files'] == read(parent_ck/'current.json')['files']
          and start['deployed_policy_sha256'] == entry_ref['sha256'] and start['replay_root_reference_and_rng_verified']
          and source_policy['postflop']['path'] == str(checkpoint/'current.npz')
          and end_policy['postflop']['path'] == str(end/'current.npz'), 'Wrong seed, rate, or source checkpoint')
    check(source_policy['preflop']['sha256'] == end_policy['preflop']['sha256'] == mix['preflop']['sha256']
          and tuple(row['iteration'] for row in mix['postflop']) == SLOTS
          and end_ref == ready['policy'] and mixture_ref == ready['mixture'], 'Wrong frozen preflop or endpoint')
    verify_export(checked_ref(end_policy['postflop']), trained['net'], mapping)
    verify_export(checked_ref(end_policy['preflop']), trained['specialist_net'], mapping)
    rows = [json.loads(line) for line in (folder/'diagnostics.jsonl').read_text().splitlines()]
    check([r['iteration'] for r in rows] == list(range(481, 505)), 'Missing or duplicate online round')
    previous = {'preflop': net_fingerprint(checked_ref(source_policy['preflop']), mapping),
                'general': net_fingerprint(checked_ref(source_policy['postflop']), mapping)}
    check(start['weights'] == previous, 'Start does not match frozen inference exports')
    selected = {r['iteration']: r for r in mix['postflop']}
    mixdir = checked_ref(mixture_ref).parent
    registered = read(mixdir/'plan.json')
    check(tuple(registered['iterations']) == SLOTS and tuple(selected) == SLOTS
          and registered['worker'] == str(folder) and registered['parent_policy_sha256'] == entry_ref['sha256']
          and mix['plan_sha256'] == digest(mixdir/'plan.json') and mix['seed'] == seed
          and digest(mixdir/mix['preflop']['file']) == source_policy['preflop']['sha256'],
          'Snapshot list, source or registered plan changed')
    seen_by_seat = [0]*6
    snapshots = {}
    for r in rows:
        t = r['iteration']
        seat = (t-1)%6
        w = r['output_weights']
        check(r['seed'] == seed and r['traverser'] == seat and r['traversals'] == 256
              and r['new_regret_rows'] > 0 and r['root_rows'] == 676
              and r['input_weights'] == previous and w['general'] != previous['general']
              and w['preflop'] == previous['preflop'] and r['frozen_preflop_probes'] == 265
              and r['frozen_preflop_max_probability_error'] == 0 and r['opening_steps'] == r['preflop_steps'] == 0
              and r['learning_rate'] == .00025 and r['general_steps'] == 768 and r['fit_draws'] == 768*512
              and len(r['fit_draws_by_street']) == len(r['fit_rows_by_street']) == 3
              and min(r['fit_draws_by_street']) > 0 and min(r['fit_rows_by_street']) > 0
              and sum(r['fit_draws_by_street']) == r['fit_draws'] and math.isfinite(r['main_loss'])
              and r['buffer_rows'] == [100000]*6 and r['reference_collection']['rows'] == 100000
              and r['reference_collection']['hands'] == 512, f'Unexpected online update at seed {seed}, step {t}')
        if t in selected:
            ref = selected[t]
            path = mixdir/ref['file']
            receipt = read(path.parent/'receipt.json')
            check(path == mixdir/f'iter-{t:05d}/current.npz' and digest(path) == receipt['export_sha256'] == ref['sha256']
                  and net_fingerprint(path, mapping) == w['general']
                  and receipt['start_sha256'] == ready['start_sha256'] and receipt['plan_sha256'] == mix['plan_sha256']
                  and receipt['iteration'] == t and receipt['export_bytes'] == path.stat().st_size,
                  'Captured policy differs from this round of this worker')
            snapshots[t] = ref['sha256']
        seen_by_seat[seat] += r['new_regret_rows']
        previous = w
    check(len(set(snapshots.values())) == 4 and snapshots[504] == end_policy['postflop']['sha256'],
          'Repeated capture or final policy mismatch')
    totals = []
    for seat, (before, after) in enumerate(zip(source['buffers'], trained['buffers'], strict=True)):
        totals.append(after['total_seen']-before['total_seen'])
        expected = [t for t in range(481, 505) if (t-1)%6 == seat]
        check(after['count'] == 100000 and np.array_equal(np.unique(after['iterations'][after['iterations']>480]), expected),
              'Latest reservoir does not include samples from all four new rounds for this seat')
    check(totals == seen_by_seat and trained['strategy_replay']['size'] == 100000
          and trained['strategy_replay']['total_seen'] == rows[-1]['reference_collection']['total_seen']
          and np.sum(trained['root_replay']['count']) - np.sum(source['root_replay']['count']) == 676*24,
          'Checkpoint counters differ from online sampling diagnostics')
    old_folder = OLD/'fixed504'/f'seed-{seed}'
    old0 = json.loads((old_folder/'diagnostics.jsonl').read_text().splitlines()[0])
    old_stage = read(OLD/'stage-ready.json')['arms']['fixed504'][key]
    verify_files({old_folder/'diagnostics.jsonl':old_stage['diagnostics_sha256']})
    common = ['input_weights','traverser','traversals','new_regret_rows','root_rows','fit_rows_by_street',
              'fit_draws_by_street','reference_collection']
    check(all(old0[k] == rows[0][k] for k in common) and old0['output_weights']['general'] != rows[0]['output_weights']['general'],
          'First matched round did not share its source/sampling counters and then diverge in fitted weights')
    return dict(seed=seed, verified_model_sha256=entry_ref['sha256'], rounds=24, traversals=24*256,
                new_regret_rows_by_seat=totals, new_regret_rows=sum(totals), postflop_updates=24*768,
                preflop_probes_per_round=265, preflop_sha256=source_policy['preflop']['sha256'],
                checkpoint_sha256=snapshots, retained_rows=[100000]*6, next_actions_reference_rows=100000,
                matched_first_round_fields=common, first_round_weights_then_diverged=True,
                historical_first_round_sample_arrays_individually_compared=False)


def training():
    plan, stage = read(STUDY/'plan.json'), read(STUDY/'stage-ready.json')
    verify_files({STUDY/'plan.json': read(STUDY/'registration.json')['plan_sha256']})
    verify_files(plan['frozen_files'])
    check(tuple(plan['seeds']) == SEEDS and plan['learning_rate'] == .00025 and plan['prior_learning_rate'] == .001
          and plan['general_steps'] == 768 and plan['main_batch'] == 512
          and tuple(plan['capture_iterations']) == SLOTS and plan['initial_iteration'] == 480 and plan['final_iteration'] == 504,
          'Training budget or registered captures differ')
    check(stage['training_complete'] and stage['iteration'] == 504, 'Training not complete')
    code = ast.parse((Path(plan['campaign'])/'tools/compact_regret.py').read_text())
    mapping = next(ast.literal_eval(node.value) for node in code.body if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == 'ARRAY_MAP' for target in node.targets))
    existing = read(OLD/'plan.json')
    fixed_update = str(ROOT/'scripts/train_poker_fixed_preflop_window.py')
    fast_traversal = str(Path(plan['campaign'])/'source/training/traverse_fast.py')
    check(existing['parent_policies'] == plan['parent_policies'] and existing['campaign'] == plan['campaign']
          and all(existing['frozen_files'][path] == plan['frozen_files'][path] for path in (fixed_update,fast_traversal)),
          'Different first-round collection implementation or parent source')
    policies = stage['policies']
    dose = read(BASE/'preflop-dose/evaluation-plan.json')['policies']
    check(policies[PARENT] == plan['parent_policies'] == dose['steps0'], 'Unconfirmed parent identity')
    seeds = {str(s):seed_training(s, plan, policies, mapping) for s in SEEDS}
    return dict(verified_utc=utc(), stage_sha256=digest(STUDY/'stage-ready.json'), training_plan_sha256=digest(STUDY/'plan.json'),
                seeds=seeds, new_selfplay_iterations_per_seed=24, total_new_regret_rows=sum(x['new_regret_rows'] for x in seeds.values()))


def private_slot(seed):
    """Derive the private hand choice directly from the registered randomization."""
    return SLOTS[random.Random(b'scheduled-postflop-choice-v1\0' + seed.to_bytes(8, 'big')).randrange(4)]


class VerifiedMixture:
    """Rebuild the deployed hand mixture without taking actions from its runtime wrapper."""

    def __init__(self, path, source):
        spec = read(path)
        directory = Path(path).parent
        self.preflop = evaluator_policy(directory/spec['preflop']['file'], source)
        self.snapshots = {row['iteration']:evaluator_policy(directory/row['file'], source) for row in spec['postflop']}
        check(tuple(self.snapshots) == SLOTS, 'Incomplete snapshot schedule')
        self.active = None

    def begin_hand(self, seed):
        self.active = self.snapshots[private_slot(seed)]

    def decide(self, state, seed):
        return (self.preflop if state['street'] == 'preflop' else self.active).decide(state, seed)


@lru_cache(maxsize=None)
def hero_policy(path, source, mixture):
    return VerifiedMixture(path, source) if mixture else evaluator_policy(path, source)


def replay(row, field, engine, bots):
    card, hero = row['seed'], row['hero']
    state = engine.call(dict(op='reset', seed=card, button=0))
    for policy in bots.values():
        if hasattr(policy, 'begin_hand'):
            policy.begin_hand(row['hand_seed'])
    trace = []
    for action in row['actions']:
        actor = state['seat_to_act']
        role = matched_role(field, card, hero, actor)
        seed = semantic_seed(card, hero, state)
        reply = (engine.call(dict(op='blueprint', seed=seed)) if role=='noregrets'
                 else bots[role].decide(state, seed))
        check(dict(reply['action'], seat=actor) == action, f'Policy action mismatch: {field} card {card} hero {hero}')
        trace.append((state['street'], actor, action['kind'], action.get('to')))
        state = engine.call(dict(op='step', action=action))
    check(state['terminal'] and state['utilities'] == row['utilities'], 'Rules engine terminal differs')
    return trace


def first_difference(left, right, hero):
    for a, b in zip(left, right):
        if a != b:
            check(a[:2] == b[:2] and a[1] == hero, 'Different actor/street or changed opponent before hero divergence')
            return a[0]
    check(left == right, 'Hand trajectories have unmatched prefix')
    return 'same_trace'


@lru_cache(maxsize=1)
def terminal_auditor():
    """Compile an isolated streaming reader against the existing pinned rules sources."""
    from build_json_adapter import COMMIT, UPSTREAM
    head = subprocess.check_output(['git','-C',str(UPSTREAM),'rev-parse','HEAD'],text=True).strip()
    clean = subprocess.check_output(['git','-C',str(UPSTREAM),'status','--porcelain'],text=True).strip()
    check(head==COMMIT and not clean,'Pinned Rust rules checkout changed')
    crate = ROOT/'.cache/noregrets-terminal-audit'
    crate.mkdir(exist_ok=True)
    name = 'poker-terminal-audit'
    reader = ROOT/'native/noregrets-json/terminal_audit.rs'
    modules = [UPSTREAM/'src'/f'{module}.rs' for module in ('cards','eval','engine')]
    header = [f'#[allow(dead_code)] #[path={json.dumps(str(file))}] mod {file.stem};' for file in modules]
    files = {'entry.rs':chr(10).join([*header,reader.read_text()]),
             'Cargo.toml':(UPSTREAM/'Cargo.toml').read_text()+f'\n[[bin]]\nname="{name}"\npath="entry.rs"\n',
             'Cargo.lock':(UPSTREAM/'Cargo.lock').read_text()}
    for filename, contents in files.items():
        target = crate/filename
        if not target.exists() or target.read_text()!=contents:
            target.write_text(contents)
    env = dict(os.environ,CARGO_HOME='/workspace/.tools/cargo',RUSTUP_HOME='/workspace/.tools/rustup',
               CARGO_TARGET_DIR=str(UPSTREAM/'target'),CARGO_BUILD_JOBS='1')
    subprocess.run([env['CARGO_HOME']+'/bin/cargo','+1.90.0','build','--offline','--locked','--release','--bin',name],
                   cwd=crate,env=env,check=True)
    return UPSTREAM/'target/release'/name


def verify_terminals(raw):
    with subprocess.Popen(['gzip','-dc',str(raw)], stdout=subprocess.PIPE) as source:
        try:
            out = subprocess.run([terminal_auditor()], stdin=source.stdout, capture_output=True, check=False)
        finally:
            source.stdout.close()
        if out.returncode:
            raise ValueError(out.stderr.decode()[:500])
        check(source.wait()==0,'Corrupt raw hand archive')
    return json.loads(out.stdout)


def read_cell(folder, job, spec, plan, parent, engine, opponents, traces, trace_stats):
    field, seed, label = (job[k] for k in ('field','seed','label'))
    ref = plan['policies'][label][str(seed)]
    start, count = spec['deal_seed'], spec['deals']
    result, prov, status = (read(folder/name) for name in ('results.json','provenance.json','status.json'))
    result_sha = digest(folder/'results.json')
    check(result['complete'] and status['phase'] == 'complete' and status['results_sha256'] == result_sha
          and result['raw_hands_verified'] and result['terminal_payoffs_replayed'] and result['verification']['terminal_payoffs_replayed']
          and result['verification']['aggregate_recomputed'] and result['verification']['chip_conservation_verified'],
          'Incomplete terminal payoff and raw-hand verification')
    raw = folder/'hands.jsonl.gz'
    expected = dict(candidate_sha256=ref['sha256'], candidate=ref['path'], field=field, deal_seed=start, deals=count,
                    hands=count*6, big_blind_chips=100, training_seed=seed, label=label)
    check(all(result.get(k)==prov.get(k)==v for k,v in expected.items()), 'Wrong registered policy, seed or card range')
    ids, roles = expected_identities(job, parent, False)
    check(result['opponents'] == prov['opponents'] == ids, 'Opponent model or rule changed')
    deps = prov['dependency_hashes']
    check(result['dependency_hashes']==deps and all(file_sha(k)==v for k,v in deps.items())
          and file_sha(ref['path'])==ref['sha256'], 'A model or implementation dependency changed')
    check(digest(raw)==result['raw_hands_sha256'] and digest(folder/'provenance.json')==result['provenance_sha256'],
          'Frozen raw or provenance checksum mismatch')
    bots = {'hero':hero_policy(ref['path'], plan['source'], label==MIX)}
    for role in roles:
        if role == 'rich96':
            key = (role, seed)
            if key not in opponents:
                rich_ref = read(RICH)['policies']['rich96'][str(seed)]
                opponents[key] = RichPolicy(checked_ref(rich_ref), None)
            bots[role] = opponents[key]
        elif role != 'noregrets':
            if role not in opponents:
                opponents[role] = Fullhouse() if role == 'public' else ReferenceOpponent(role)
            bots[role] = opponents[role]
    sums = np.zeros(count, dtype=np.int64)
    snapshot_counts = Counter()
    hands = actions = replayed = replayed_decisions = 0
    with gzip.open(raw, 'rt') as stream:
        for line in stream:
            row = json.loads(line)
            n, hero = divmod(hands, 6)
            card, rng = start+n, hand_seed(start+n, hero)
            check(n < count and (row['deal'],row['seed'],row['hero'],row['hand_seed']) == (n,card,hero,rng),
                  'Missing, repeated, or incorrect card/seat')
            if label == MIX:
                slot = private_slot(rng)
                check(row['snapshot_iteration'] == slot, 'Incorrect hand-private mixture selection')
                snapshot_counts[slot] += 1
            else:
                check('snapshot_iteration' not in row, 'Snapshot assigned to a non-mixture policy')
            pay = row['utilities']
            check(len(pay)==6 and all(type(v) is int for v in pay) and sum(pay)==0 and type(row['chips']) is int
                  and row['chips']==pay[hero] and len(row['actions'])==row['steps'] and 0<row['steps']<=512,
                  'Invalid hand utility or action count')
            sums[n] += row['chips']
            if n % SAMPLE_STRIDE == 0 or n == count-1:
                trace = replay(row, field, engine, bots)
                if label == PARENT:
                    traces[(n,hero)] = (trace, row['chips'])
                elif label in (MIX, END):
                    before, original_chips = traces[(n,hero)]
                    street = first_difference(before, trace, hero)
                    change = row['chips']-original_chips
                    check(street != 'preflop' and (street != 'same_trace' or change==0),
                          'Different preflop play or different payoffs on the same trajectory')
                    trace_stats[street]['hands'] += 1
                    trace_stats[street]['delta_chips'] += change
                replayed += 1
                replayed_decisions += row['steps']
            actions += row['steps']
            hands += 1
    check(hands==count*6 and actions==result['verification']['concrete_decisions_replayed'],
          'Incomplete hand or decision count')
    check(verify_terminals(raw)==dict(hands=hands,decisions=actions), 'Streaming terminal audit count differs')
    blocks = sums/6
    close_to(blocks, result['deal_blocks'], 'Incorrect six-seat means')
    close_to(blocks.mean(), result['bb_per_100'], 'Incorrect bb/100')
    check({str(k):v for k,v in snapshot_counts.items()} == result['snapshot_hand_counts'], 'Incorrect per-cell hand selections')
    check(prov['evaluation_plan_sha256']==file_sha(folder.parent.parent/'evaluation-plan.json'), 'Wrong cell plan')
    if label == MIX:
        check(set(snapshot_counts)==set(SLOTS), 'One or more frozen snapshots were never selected')
    return blocks, dict(hands=hands, decisions=actions, policy_replayed_hands=replayed,
                        policy_replayed_decisions=replayed_decisions, selected_hand_counts=dict(snapshot_counts),
                        raw_sha256=result['raw_hands_sha256'], result_sha256=result_sha)


file_sha = lru_cache(maxsize=None)(digest)


def paired_stats(left, right, z):
    check(left.shape == right.shape and left.ndim == 2 and left.shape[0] == len(SEEDS)
          and left.shape[1] >= 2 and np.isfinite(left).all() and np.isfinite(right).all(),
          'Expected matched fixed-seed, original-card means')
    by_seed = right-left
    values = by_seed.mean(axis=0)
    return dict(mean=float(values.mean()), interval=interval(values,z), fixed_seed_changes=by_seed.mean(axis=1).tolist())


def audit_pair(summary, groups, data, left, right):
    check(summary['baseline'] == left and summary['candidate']==right, 'Wrong pair reported')
    z = NormalDist().inv_cdf(1-.05/(2*3*len(groups)))
    check(set(summary['fields']) == set(groups) and set(summary['roles']) == {s['role'] for s in groups.values()}
          and summary['training_seeds'] == list(SEEDS) and summary['units'] == 'bb/100',
          'Missing fields, roles or fixed seeds')
    scores = {}
    for field, policies in data.items():
        first, second = policies[left], policies[right]
        expected = paired_stats(first, second, z)
        reported = summary['fields'][field]
        for label, matrix in [(left,first),(right,second)]:
            close_to(reported[label]['mean'], matrix.mean(), 'Unpaired absolute score')
            close_to([reported[label]['lower'],reported[label]['upper']], interval(matrix.mean(axis=0),z),
                     'Unpaired absolute confidence interval')
        close_to(reported['paired_change']['mean'], expected['mean'], 'Changed paired difference')
        close_to([reported['paired_change']['lower'], reported['paired_change']['upper']], expected['interval'],
                 'Wrong paired statistical unit or confidence interval')
        scores[field] = dict(absolute={left:float(first.mean()),right:float(second.mean())}, change=expected)
    for role, group in summary['roles'].items():
        fields = [f for f,s in groups.items() if s['role']==role]
        families = Counter(groups[f]['family'] for f in fields)
        weights = {f:1/(len(families)*families[groups[f]['family']]) for f in fields}
        check(set(weights)==set(group['field_weights']), 'Wrong declared families')
        for f in fields:
            close_to(weights[f],group['field_weights'][f],'Wrong field weights')
        expected_uncapped = {
            'mean':sum(w*scores[f]['change']['mean'] for f,w in weights.items()),
            'lower':sum(w*scores[f]['change']['interval'][0] for f,w in weights.items()),
            'upper':sum(w*scores[f]['change']['interval'][1] for f,w in weights.items())}
        for key, value in expected_uncapped.items():
            close_to(value,group['uncapped_family_balanced_change_bb100'][key],'Wrong uncapped group difference')
        for label in (left,right):
            check(set(group['breadth'][label]['positive_fields'])=={f for f in fields if scores[f]['absolute'][label]>0},
                  'Wrong profitable field count')
            check(group['breadth'][label]['weakest_field_by_point_estimate'] == min(fields,key=lambda f:scores[f]['absolute'][label]),
                  'Incorrect weakest opponent')
        for cap in (50,100,200):
            expected = {label:sum(w*min(cap,scores[f]['absolute'][label]) for f,w in weights.items()) for label in (left,right)}
            report = group['family_balanced_capped_bb100'][str(cap)]
            for label,v in expected.items():
                close_to(v, report[label], 'Wrong capped family-weighted score')
            close_to(expected[right]-expected[left], report['paired_change']['mean'], 'Wrong capped difference')
            # Each clipped difference respects both monotonicity and the unit Lipschitz bound.
            capped_low = capped_high = 0.0
            for field, weight in weights.items():
                old = interval(data[field][left].mean(axis=0),z)
                new = interval(data[field][right].mean(axis=0),z)
                low, high = scores[field]['change']['interval']
                capped_low += weight * max(min(0,low), min(cap,new[0])-min(cap,old[1]))
                capped_high += weight * min(max(0,high), min(cap,new[1])-min(cap,old[0]))
            close_to([capped_low,capped_high], [report['paired_change']['lower'],report['paired_change']['upper']],
                     'Incorrect capped confidence interval')
    return scores


def audit():
    training_result = training()
    population = STUDY/'population'
    ref = read(STUDY/'evaluation-design-reference.json')
    registration = read(population/'registration.json')
    plan_path = population/'evaluation-plan.json'
    plan, summary = read(plan_path), read(population/'summary.json')
    cfg = ROOT/'configs/poker-postflop-averaging-evaluation-20261120.json'
    hashes = {cfg: '32364b5cb013d692b62fbc975c7cb261d70b878712bc7bbe0a58681ac546874b',
              STUDY/'registration.json': ref['training_registration_sha256'],
              STUDY/'plan.json': ref['training_plan_sha256'], population/'design.json': registration['design_sha256'],
              plan_path: read(population/'bound.json')['evaluation_plan_sha256'],
              BASE/'preflop-dose/evaluation-plan.json': '642491686308caeaef39f80e90277f8e77d437a6f80b37cae1fa4b47a8a6c4d6'}
    verify_files(hashes)
    design = read(population/'design.json')
    groups = plan['fields']
    stage = read(STUDY/'stage-ready.json')
    parent = read(BASE/'preflop-dose/evaluation-plan.json')
    expected_policies = {**stage['policies'], DOSE: parent['policies'][DOSE]}
    check(plan['policies'] == expected_policies and plan['policies'][PARENT] == parent['policies']['steps0']
          and plan['seeds'] == list(SEEDS) and read(cfg) == design['config'] and design['fields'] == groups
          and ref['before_evaluation'] and registration['config_sha256']==ref['evaluation_config_sha256']==digest(cfg)
          and plan['stage_sha256']==training_result['stage_sha256'] and summary['plan_sha256']==digest(plan_path)
          and summary['complete'] and not summary['model_promoted'], 'Study or result registration mismatch')
    verify_files(plan['frozen_files'])
    prior = [item['range'] for item in prior_card_ranges(STUDY).values()]
    seen = set(FIELD_CARDS.values())
    for start, count in seen:
        check(all(start+count<=lo or hi<=start for lo,hi in prior), 'Card range reused from another study')
    check(set(groups)==set(FIELD_CARDS) and all((spec['deal_seed'],spec['deals']) == FIELD_CARDS[f] for f,spec in groups.items())
          and sum(n for _,n in seen)==summary['unique_original_card_blocks']==24000, 'Unmatched or reused cards')
    expected_jobs = [dict(id=f'{f}-seed-{s}-{label}', seed=s, field=f, label=label) for f in groups for s in SEEDS
                     for label in ([PARENT,MIX,END,DOSE] if f=='public' else [PARENT,MIX,END])]
    check(plan['jobs']==design['jobs']==expected_jobs and len(plan['jobs'])==summary['completed_cells']==75,
          'Changed cell assignment or missing schedule')
    reg = read(ROOT/'artifacts/selfplay/current.json')
    bridge_model = ROOT/reg['model']
    public = Fullhouse()
    check(reg['sha256']==parent['opponents']['public']['noregrets']==digest(bridge_model)
          and public.digest==parent['opponents']['public']['public'], 'Opponent checkpoint differs')
    proofs, data, snapshot_counts, sampled_traces = {}, {}, {}, {}
    opponents = {'public':public}
    with ExitStack() as engines:
        regular = engines.enter_context(closing(Bridge()))
        blueprint = engines.enter_context(closing(Bridge(bridge_model)))
        for field, spec in groups.items():
            need_blueprint = field in ('noregrets','mixed','varied')
            matrix, field_traces = {}, {}
            for seed in SEEDS:
                traces = {}
                for job in expected_jobs:
                    if job['field']!=field or job['seed']!=seed:
                        continue
                    name, label = job['id'], job['label']
                    tracked = field_traces.setdefault(label,{s:Counter() for s in ('flop','turn','river','same_trace')})
                    blocks, proof = read_cell(population/'evaluation'/name, job, spec, plan, parent,
                                               blueprint if need_blueprint else regular, opponents, traces, tracked)
                    matrix.setdefault(label,[]).append(blocks)
                    proofs[name] = proof
                    if label==MIX:
                        snapshot_counts[name] = {str(k):v for k,v in proof['selected_hand_counts'].items()}
            data[field] = {label:np.asarray(rows) for label,rows in matrix.items()}
            sampled_traces[field] = {label:field_traces[label] for label in (MIX,END)}
    check({k:v['result_sha256'] for k,v in proofs.items()} == summary['cell_result_sha256']
          and snapshot_counts==summary['snapshot_hand_counts'], 'Different raw files or assigned snapshots')
    total = sum(p['hands'] for p in proofs.values())
    check(total==plan['planned_hands']==summary['completed_hands']==2196000, 'Incomplete hand count')
    comparisons = {'primary': (summary['primary'],PARENT,MIX),
                   'endpoint_vs_parent':(summary['diagnostics']['current_vs_480'],PARENT,END),
                   'mixture_vs_endpoint':(summary['diagnostics']['mixture_vs_current'],END,MIX)}
    scores = {name:audit_pair(record, groups, data, left, right) for name,(record,left,right) in comparisons.items()}
    public_results = {}
    for left,right in ((PARENT,MIX),(PARENT,END),(END,MIX),(DOSE,MIX),(PARENT,DOSE)):
        name = f'{right}_minus_{left}'
        old,new = data['public'][left],data['public'][right]
        observed = paired_stats(old, new, NormalDist().inv_cdf(.975))
        reported = summary['public_comparisons'][name]
        check((reported['baseline'],reported['candidate'])==(left,right),'Wrong public diagnostic')
        for key,value in [('baseline_bb100',old.mean()),('candidate_bb100',new.mean()),('mean',observed['mean']),
                          ('paired_ci95',observed['interval']),('by_seed',observed['fixed_seed_changes'])]:
            close_to(reported[key],value,'Different public head-to-head result')
        public_results[name] = observed
    return dict(verified_utc=utc(), study=str(STUDY), summary_sha256=digest(population/'summary.json'),
                training=training_result, evaluation_plan_sha256=digest(plan_path), cells=len(proofs),
                hands=total, distinct_original_cards=24000, cells_verified_from_raw=proofs,
                independent_rules_terminal_replay_hands=total,
                terminal_auditor_binary_sha256=digest(terminal_auditor()),
                terminal_auditor_source_sha256=digest(ROOT/'native/noregrets-json/terminal_audit.rs'),
                independent_policy_and_terminal_replay_stride=SAMPLE_STRIDE,
                independent_policy_and_terminal_replay_hands=sum(r['policy_replayed_hands'] for r in proofs.values()),
                independent_policy_replay_decisions=sum(r['policy_replayed_decisions'] for r in proofs.values()),
                primary_and_diagnostic_results=scores, public_nominal_comparisons=public_results,
                paired_sample_first_different_action_by_street=sampled_traces,
                statistical_unit='original card block after averaging six seats and three fixed model seeds')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('training','all'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = training() if args.phase=='training' else audit()
    if args.output:
        atomic_json(args.output, result)
    else:
        print(json.dumps(result, indent=2, allow_nan=False))
