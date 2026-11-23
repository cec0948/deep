"""Bind the matched-control, historical opponent pool, and fresh evaluation ranges."""
from pathlib import Path

from controlled_poker_refit import read, utc, verify_files
from fullhouse_v3_common import ROOT, atomic_json, digest
from run_compact_refit_probe import check_card_range
from run_poker_candidate_confirmation import prior_card_ranges

REL = Path('preflop-followup/480-v1/opponent-diversity-20261123')
STUDY = ROOT/'artifacts'/REL
REPORT = ROOT/'reports'/REL
CONFIG = ROOT/'configs/poker-opponent-diversity-20261123.json'
CONTROL = STUDY.parent/'postflop-averaging-20261120'
BOUNDED = ROOT/'artifacts/bounded-continuation/20261113T014811152621Z'
CONTINUATION = ROOT/'artifacts/average-continuation/20261117T032100Z'
EXTENSION = ROOT/'artifacts/average-extension/20261117T063900Z'


def register():
    cfg = read(CONFIG)
    frozen = {
        CONTROL/'stage-ready.json':'288ec2d5f9068221b5922cfeac29301f226e030734e67b15a4399748224d071e',
        CONTROL/'population/summary.json':'52a98ce8ba93ac612c48b679a6cd473685a96730db1c1c0bcd4f83c90bb89448',
        CONTROL/'population/evaluation-plan.json':'87927530b0cfbd8e049769fef29dcdc8a02640869ef189799c1e91e676c6ae07',
        BOUNDED/'stage-ready.json':'f8a85f6f947658ac075e73a8aaff1235ebab7f136c7d6892750b2239a50f6a2a',
        CONTINUATION/'stage-ready.json':'3a8563fccf7acf7518c1899b8c89ce47412a83021a054764dcf26a781861ea65',
        EXTENSION/'stage-ready.json':'b4a01a4633d0a94fea9d3e5818c5be7ff95f054faef708177141640fc5d79f39',
        ROOT/'.cache/fullhouse-bot/bots/adversarial/solver_like.py':cfg['new_opponent_code_sha256'],
    }
    verify_files(frozen)
    control = read(CONTROL/'stage-ready.json')['policies']
    historical = {}
    for seed in cfg['seeds']:
        sid = str(seed)
        assets = {}
        for step, folder in [(300, BOUNDED), (360, BOUNDED), (420, CONTINUATION), (480, EXTENSION)]:
            stage = read(folder/'stage-ready.json')
            checksum = (stage['checkpoints'][str(step)]['models'][sid] if step < 420 else
                        stage['seeds'][sid]['files']['current.npz'])
            source = folder/'campaign'/f'seed-{seed}'/'checkpoints'/f'iter-{step:05d}'/'current.npz'
            frozen[source] = checksum
            assets[str(step)] = {'path':str(source), 'sha256':checksum}
        frozen[CONTINUATION/'campaign'/f'seed-{seed}'/'parent360.npz'] = assets['360']['sha256']
        frozen[EXTENSION/'campaign'/f'seed-{seed}'/'parent420.npz'] = assets['420']['sha256']
        for label in ['specialist480','mixture504']:
            ref = control[label][sid]
            frozen[Path(ref['path'])] = ref['sha256']
        historical[sid] = assets
        if read(control['specialist480'][sid]['path'])['postflop']['sha256'] != assets['480']['sha256']:
            raise ValueError('The 480 starting policy and historical lineage differ')
    verify_files(frozen)
    previous = {name:entry['range'] for name,entry in prior_card_ranges(STUDY).items()}
    checked = len(previous)
    for label,(start,count) in cfg['cards'].items():
        check_card_range(start,count,previous)
        previous['current-'+label] = [start,start+count]
    STUDY.mkdir(parents=True,exist_ok=True)
    REPORT.mkdir(parents=True,exist_ok=True)
    registration = STUDY/'registration.json'
    with (STUDY/'registration.lock').open('x'):
        pass
    for name,(start,count) in cfg['cards'].items():
        folder = STUDY/'reservations'/name
        folder.mkdir(parents=True,exist_ok=True)
        atomic_json(folder/'config.json',dict(evaluation_deal_seed=start,evaluation_deals=count))
    payload = dict(created_utc=utc(),config_sha256=digest(CONFIG),control_training_plan_sha256=digest(CONTROL/'plan.json'),
                   checked_prior_card_ranges=checked, fresh_evaluation_cards=cfg['cards'], historical_policies=historical,
                   frozen_files={str(path):sha for path,sha in frozen.items()})
    atomic_json(registration,payload)
    atomic_json(REPORT/'registration.json',payload)


if __name__=='__main__':
    register()
