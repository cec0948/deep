"""Lock a continuation to the last verified state, fresh cards and fixed study budget."""
from pathlib import Path

from controlled_poker_refit import read, utc, verify_files
from fullhouse_v3_common import ROOT, atomic_json, digest
from run_compact_refit_probe import check_card_range
from run_poker_candidate_confirmation import prior_card_ranges


REL = Path('preflop-followup/480-v1/pool-extension-528-20261126')
STUDY = ROOT/'artifacts'/REL
REPORT = ROOT/'reports'/REL
CONFIG = ROOT/'configs/poker-pool-extension-528-20261126.json'
SOURCE = STUDY.parent/'opponent-diversity-20261123'


def register():
    cfg = read(CONFIG)
    stage_path = SOURCE/'stage-ready.json'
    stage = read(stage_path)
    audit = read(SOURCE/'training-audit.json')
    if not stage['training_complete'] or stage['iteration'] != 504 or audit['stage_sha256'] != cfg['source_stage_sha256']:
        raise ValueError('Incorrect continuation source')
    assets = {stage_path:cfg['source_stage_sha256'], SOURCE/'training-plan.json':audit['training_plan_sha256']}
    for seed in cfg['seeds']:
        sid = str(seed)
        run = SOURCE/'pool504'/f'seed-{seed}'
        current = read(run/'current.json')
        expected = stage['arms']['pool504'][sid]['files']
        if current['iteration'] != 504 or current['files'] != expected:
            raise ValueError('Source checkpoint registry differs from verified stage')
        assets.update({run/'checkpoints'/current['directory']/n:h for n,h in expected.items()})
        assets[run/'current.json'] = digest(run/'current.json')
        for policies in stage['policies'].values():
            model = policies[sid]
            assets[Path(model['path'])] = model['sha256']
    prior = read(SOURCE/'training-plan.json')['historical_policies']
    for stages in prior.values():
        for ref in stages.values():
            assets[Path(ref['path'])] = ref['sha256']
    verify_files(assets)
    seen = {name:v['range'] for name,v in prior_card_ranges(STUDY).items()}
    old_ranges = len(seen)
    for name,(start,count) in cfg['cards'].items():
        check_card_range(start,count,seen)
        seen[name] = [start,start+count]
    STUDY.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    with (STUDY/'registration.lock').open('x'):
        pass
    for name,(start,count) in cfg['cards'].items():
        folder = STUDY/'reservations'/name
        folder.mkdir(parents=True, exist_ok=True)
        atomic_json(folder/'config.json', dict(evaluation_deal_seed=start,evaluation_deals=count))
    receipt = dict(created_utc=utc(), config_sha256=digest(CONFIG), source_stage_sha256=digest(stage_path),
                   checked_prior_ranges=old_ranges, evaluation_cards=cfg['cards'], fixed_baseline='specialist480',
                   learning_source_iteration=504, final_iteration=528, training_updates_per_seed=24)
    atomic_json(STUDY/'registration.json',dict(receipt,frozen_files={str(p):h for p,h in assets.items()}))
    atomic_json(REPORT/'registration.json',receipt)


if __name__ == '__main__':
    register()
