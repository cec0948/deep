#!/usr/bin/env python3
"""Bounded staged training using the existing NoRegrets CLI, no paid services."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / '.cache/noregrets'

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--through',type=int,default=25000,help='Configured milestone to finish at')
    p.add_argument('--run-dir',type=Path,help='Resume a previous campaign directory')
    a=p.parse_args()
    cfg=json.loads((ROOT/'configs/selfplay-pilot.json').read_text())
    if a.through not in cfg['milestones']:p.error('--through must be a configured milestone')
    head=subprocess.check_output(['git','-C',str(UPSTREAM),'rev-parse','HEAD'],text=True).strip()
    dirty=subprocess.check_output(['git','-C',str(UPSTREAM),'status','--porcelain'],text=True).strip()
    if head!=cfg['upstream_commit'] or dirty:raise SystemExit('Upstream source changed')
    binary=UPSTREAM/'target/release/pluribus'
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out=(a.run_dir or ROOT/'artifacts/selfplay'/stamp).resolve()
    out.mkdir(parents=True,exist_ok=True)
    manifest=out/'campaign.json'
    if manifest.exists():
        state=json.loads(manifest.read_text())
        if state['config']!=cfg or state['binary_sha256']!=sha(binary):raise SystemExit('Campaign configuration or binary changed')
    else:
        if a.run_dir:raise SystemExit('Specified campaign manifest does not exist')
        state=dict(config=cfg,binary_sha256=sha(binary),stages=[],purpose='research pilot; strength unverified')
    def save():
        temporary=out/'campaign.json.tmp'
        temporary.write_text(json.dumps(state,indent=2)+'\n')
        temporary.replace(manifest)
    save()
    print(f'Campaign: {out}',flush=True)
    env=os.environ.copy();env['RAYON_NUM_THREADS']=str(cfg['threads'])
    for target in cfg['milestones']:
        if target>a.through:break
        previous=state['stages'][-1] if state['stages'] else None
        if previous and previous['iterations']>=target:continue
        if previous and (sha(out/previous['checkpoint'])!=previous['checkpoint_sha256']):raise SystemExit('Checkpoint integrity mismatch')
        start=previous['iterations'] if previous else 0
        bp=out/f'blueprint-{target}.bin';ckpt=out/f'checkpoint-{target}.bin';log=out/f'train-{target}.log'
        if any(x.exists() for x in (bp,ckpt,log)):raise SystemExit('Preserving previous stage files; use a fresh campaign after a failed stage')
        command=[str(binary),'train','--players','6','--stack','10000','--threads',str(cfg['threads']),
                 '--train-seed',str(cfg['seed']),'--menu',cfg['menu'],'--iters',str(target-start),
                 '--out',str(bp),'--checkpoint',str(ckpt)]
        if previous:command+=['--resume',str(out/previous['checkpoint'])]
        else:command+=['--buckets',str(cfg['buckets']),'--rollouts',str(cfg['rollouts']),
                      '--runouts',str(cfg['runouts']),'--kmeans-samples',str(cfg['kmeans_samples'])]
        started=time.monotonic();peak=0;reason=None
        print(f'Training {start} -> {target}',flush=True)
        with log.open('w') as stream:
            process=subprocess.Popen(command,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            while process.poll() is None:
                try:
                    text=Path(f'/proc/{process.pid}/status').read_text()
                    rss=next((int(line.split()[1])*1024 for line in text.splitlines() if line.startswith('VmRSS:')),0)
                    peak=max(peak,rss)
                except FileNotFoundError:pass
                if peak>cfg['max_rss_gib']*1024**3:reason='rss_limit'
                if time.monotonic()-started>cfg['stage_timeout_seconds']:reason='time_limit'
                if reason:
                    os.killpg(process.pid,signal.SIGTERM);process.wait();break
                time.sleep(0.5)
        elapsed=time.monotonic()-started
        if process.returncode or reason:
            state['failure']=dict(target=target,reason=reason,returncode=process.returncode,elapsed_seconds=elapsed,command=command)
            save();raise SystemExit(f'Training stopped: {state["failure"]}')
        inspected=subprocess.check_output([str(binary),'inspect','--blueprint',str(bp)],text=True)
        (out/f'inspect-{target}.log').write_text(inspected)
        if f'trained iterations: {target}\n' not in inspected or 'players: 6\n' not in inspected:raise SystemExit('Unexpected training outcome')
        stage=dict(iterations=target,command=command,elapsed_seconds=round(elapsed,3),peak_rss_bytes=peak,
                   blueprint=bp.name,blueprint_sha256=sha(bp),blueprint_bytes=bp.stat().st_size,
                   checkpoint=ckpt.name,checkpoint_sha256=sha(ckpt),checkpoint_bytes=ckpt.stat().st_size)
        state['stages'].append(stage);save();print(json.dumps(stage),flush=True)
    print(f'Manifest: {manifest}',flush=True)

if __name__=='__main__':main()
