"""Independently verify the matched postflop continuations at observed training iterations."""
import ast
from contextlib import closing
import math
from pathlib import Path
from statistics import NormalDist, mean, variance

import numpy as np

from audit_poker_flop_diagnostic import (
    OLD, SLOTS, STUDY, close, frozen_policies, json_lines, load_training_engine, locked_inputs,
    original_cards, read_json, replay, replay_policy_branch, replicate_deck, require, timestamp,
)
from audit_postflop_averaging import net_fingerprint
from fullhouse_runtime import Bridge
from fullhouse_v3_common import atomic_json, digest
from poker_multivalued_turn import Rules

OUT = STUDY / 'selfplay-window'
PREVIOUS_AUDIT = 'd4a715b2d32655a84a4ec2351dc6bdc09b60ecf261d203c5f707d8d19662931b'
PREVIOUS_SUMMARY = '11e82ac96c9d05701b40440a296c76c0f6c782aebac1f361076a93ba7f06c056'
ADDED = ('self486', 'self492', 'self498')
TRAINING_FIELDS = ('self480', *ADDED)


def training_opponents(plan, seed):
    training_plan = read_json(OLD / 'plan.json')
    source = Path(training_plan['campaign']) / 'tools/compact_regret.py'
    keys = next(ast.literal_eval(node.value) for node in ast.parse(source.read_text()).body if
                isinstance(node, ast.Assign) and any(getattr(t, 'id','') == 'ARRAY_MAP' for t in node.targets))
    refs = plan['policies']
    parent = read_json(Path(refs['specialist480'][str(seed)]['path']))
    mix = Path(refs['mixture504'][str(seed)]['path'])
    desc = read_json(mix)
    common_preflop = Path(parent['preflop']['path'])
    require(digest(common_preflop) == parent['preflop']['sha256'] == desc['preflop']['sha256']
            == digest(mix.parent/desc['preflop']['file']), 'Different preflop specialist')
    pre_hash = net_fingerprint(common_preflop, keys)
    files = {480:Path(parent['postflop']['path'])}
    for ref in desc['postflop']:
        if ref['iteration'] in (486,492,498):
            model = mix.parent/ref['file']
            require(digest(model) == ref['sha256'], 'Wrong opponent snapshot')
            files[ref['iteration']] = model
    require(tuple(sorted(files)) == (480,486,492,498), 'Incomplete cohort of training opponents')
    worker = OLD/'slow504'/f'seed-{seed}'
    ledger = worker/'diagnostics.jsonl'
    require(digest(ledger) == read_json(worker/'ready.json')['diagnostics_sha256'], 'Changed training diagnostics')
    rows = {r['iteration']:r for r in json_lines(ledger)}
    verified = {}
    for iteration, path in files.items():
        general = net_fingerprint(path, keys)
        require(rows[iteration+1]['input_weights'] == dict(preflop=pre_hash,general=general),
                'Weights were not the ones used for traversals in the next iteration')
        if iteration > 480:
            require(general == rows[iteration]['output_weights']['general'], 'Wrong trained output')
        verified[iteration] = dict(preflop_sha256=digest(common_preflop), postflop_sha256=digest(path),
                                   verified_input_used_at=iteration+1)
    policies=frozen_policies(plan,seed)
    return {'self'+str(i):policies[str(i)] for i in files}, verified


def outcomes(primary, second, states, expected_repeats):
    values, effects = {}, {}
    for file, fields in ((primary, ('self480','public')), (second, ADDED)):
        for row in json_lines(file):
            field = row['lineup']
            if file == primary and field == 'tight-value':
                continue
            ident = row['state_id']
            state = states[ident]
            key = ident,field
            require(key not in values and field in fields and row['repeats'] == expected_repeats
                    and [row[k] for k in ('legal_actions','seed','hero','deal_seed')] ==
                        [state[k] for k in ('legal_actions','seed','hero','deal_seed')], 'Crossed or duplicate branch')
            raw = row['action_payoffs_chips']
            probs = np.array([state['policies'][s]['probabilities'] for s in SLOTS])
            probs /= probs.sum(axis=1,keepdims=True)
            require(len(raw) == probs.shape[1] and all(
                (a is None and not p.any()) or (p.any() and isinstance(a,list) and len(a)==expected_repeats
                                                and all(type(v) is int and -10000<=v<=50000 for v in a))
                for p,a in zip(probs.T,raw)),'Unexpected action/replicate support')
            values[key] = raw
            by_policy = [math.fsum(chips * policy[i] for i,ls in enumerate(raw) if ls is not None for chips in ls)
                         / (100*expected_repeats) for policy in probs]
            effects[key] = mean(by_policy[1:])-by_policy[0]
    require(len(values) == len(states)*5, 'Missing matched valuations')
    return values,effects


def audit_seed(cfg, plan, seed, previous):
    original, folder = (path/f'seed-{seed}' for path in (STUDY,OUT))
    state_file = original/'states.jsonl'
    require(digest(state_file) == previous['states_sha256']
            and digest(original/'values.jsonl.gz') == previous['values_sha256'], 'Reused primary input changed')
    states = {s['state_id']:s for s in json_lines(state_file)}
    repeated = cfg['paired_repetitions']
    value_file, trace_file = folder/'values.jsonl.gz', folder/'traces.jsonl.gz'
    cells,effects = outcomes(original/'values.jsonl.gz', value_file, states, repeated)
    # Audit the three added lineups here; the registered prior receipt covers the
    # fourth self-play condition and independent Fullhouse as well as all original roots.
    wanted = {(ident,f,i) for (ident,f),outs in cells.items() if f in ADDED
              for i,v in enumerate(outs) if v is not None}
    picked = set(list(states)[::64])
    original_decks = {}
    bots,opponents = training_opponents(plan,seed)
    game = load_training_engine()
    branch_replays = policy_replays = 0
    with closing(Bridge()) as bridge, closing(Rules()) as oracle:
        for trace in json_lines(trace_file):
            ident,field,idx = (trace[k] for k in ('state_id','lineup','action_index'))
            key = ident,field,idx
            require(key in wanted, 'Unexpected repeated or unsupported historical-opponent arm')
            wanted.remove(key)
            row = states[ident]
            deal,hero = row['deal_seed'],row['hero']
            if deal not in original_decks:
                original_decks[deal] = original_cards(bridge,deal)
            fixed = original_decks[deal]
            require(trace['replicate'] == 0 and trace['deck'] == replicate_deck(fixed,deal,0)
                    and trace['seed'] == seed and trace['deal_seed'] == deal and trace['hero'] == hero
                    and trace['button'] == 0 and trace['prefix'] == row['prefix']
                    and trace['actions'][0] == dict(row['legal_actions'][idx],seat=hero), 'Wrong action or pairing')
            require(replay(game,trace) == cells[ident,field][idx][0], 'First settlement differs from training engine')
            branch_replays += 1
            if ident in picked:
                for r in range(repeated):
                    utilities,history,_ = replay_policy_branch(oracle,replicate_deck(fixed,deal,r),row,
                                            row['legal_actions'][idx],field,bots,r,r==0)
                    require(utilities[hero] == cells[ident,field][idx][r] and
                            (r>0 or (history == trace['actions'] and utilities == trace['utilities'])),
                            'Different training-opponent decision, RNG draw, or terminal settlement')
                    policy_replays += 1
    require(not wanted and len(original_decks)==len(states),'Incomplete matched training-opponent branches')
    returns = []
    for ident in states:
        train = mean(effects[ident,f] for f in TRAINING_FIELDS)
        returns.append((train,effects[ident,'public']-train))
    return returns,dict(training_opponents=opponents, python_training_engine_rep0_branches=branch_replays,
                        reproduced_added_policy_branches=policy_replays, all_added_branches=branch_replays*repeated,
                        states_sha256=digest(state_file), values_sha256=digest(value_file), traces_sha256=digest(trace_file))


def audit():
    cfg, plan = locked_inputs()
    require(digest(STUDY/'final-audit.json') == PREVIOUS_AUDIT and digest(STUDY/'summary.json') == PREVIOUS_SUMMARY,
            'Original comparison no longer matches its verified result')
    registration_file = OUT/'registration.json'
    reg = read_json(registration_file)
    require(reg['primary_diagnostic_already_scored'] and tuple(reg['fields_new']) == ADDED and
            reg['per_state_window_weights'] == [.25]*4 and reg['additional_deals'] == 0,'Wrong supplementary design')
    for path, value in reg['frozen_files'].items():
        require(digest(path)==value, 'Changed frozen source or raw input')
    stats = read_json(OUT/'summary.json')
    require(stats['registration_sha256']==digest(registration_file), 'Wrong supplementary registration')
    previous = read_json(STUDY/'final-audit.json')
    record, gains = {}, {}
    for seed in cfg['training_seeds']:
        folder = OUT/f'seed-{seed}'
        receipt = read_json(folder/'receipt.json')
        gains[seed], details = audit_seed(cfg,plan,seed,previous['by_seed'][str(seed)])
        require(receipt==stats['receipts'][str(seed)] and all(receipt[k]==details[k] for k in
                ('states_sha256','values_sha256','traces_sha256')) and receipt['states']==len(gains[seed])
                and receipt['trace_count']==details['python_training_engine_rep0_branches']
                and receipt['terminal_branches']==details['all_added_branches'],'Receipt mismatch')
        for iteration,proof in details['training_opponents'].items():
            frozen=reg['frozen_cycles_by_seed'][str(seed)][str(iteration)]
            require(proof['postflop_sha256']==frozen['postflop_export_sha256'] and
                    proof['verified_input_used_at']==frozen['start_round'],'Different captured opponent')
        record[str(seed)] = details
    subtotal = sum(d['all_added_branches'] for d in record.values())
    require(subtotal == stats['additional_terminal_branches'] and
            sum(len(v) for v in gains.values()) == stats['unique_original_deals'] == 3072,'Incomplete branch counts')
    names = ['mixture_minus_480_on_window','public_minus_window_on_mixture_change']
    require(set(stats['contrasts'])==set(names),'Wrong registered contrasts')
    estimates={}
    for idx,name in enumerate(names):
        values = {s:[r[idx] for r in rows] for s,rows in gains.items()}
        means = {k:mean(v) for k,v in values.items()}
        m = mean(means.values())
        se = math.sqrt(sum(variance(v)/len(v) for v in values.values()))/len(means)
        row = stats['contrasts'][name]
        close([m,se],[row['mean_bb'],row['cluster_se_bb']])
        for k,val in means.items():
            close(val,row['mean_bb_by_seed'][str(k)])
        for key,prob in [('paired_95_bb',.975),('simultaneous_95_bb',.9875)]:
            z = NormalDist().inv_cdf(prob)
            close([m-z*se,m+z*se],row[key])
        estimates[name] = dict(mean_bb=m,cluster_se_bb=se)
    return dict(verified_utc=timestamp(), primary_audit_sha256=PREVIOUS_AUDIT,
                registration_sha256=digest(registration_file), summary_sha256=digest(OUT/'summary.json'),
                script_sha256=digest(__file__), unique_original_deals=stats['unique_original_deals'],
                added_terminal_branches_reaggregated=subtotal,python_replayed_first_repeats=sum(
                    d['python_training_engine_rep0_branches'] for d in record.values()),
                eight_repeat_policy_replays=sum(d['reproduced_added_policy_branches'] for d in record.values()),
                independently_recomputed=estimates, by_seed=record,
                scope='Four directly observed training-window opponents at 480/486/492/498; '
                      'the forced single flop action always has a common frozen hero continuation.')


if __name__ == '__main__':
    atomic_json(OUT/'final-audit.json',audit())
