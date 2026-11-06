#!/usr/bin/env python3
"""Build a transport adapter over immutable, pinned NoRegrets source."""
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / '.cache/noregrets'
COMMIT = '757f7692738069522195d2b486eec60a8b010c0c'

def main():
    head = subprocess.check_output(['git','-C',str(UPSTREAM),'rev-parse','HEAD'],text=True).strip()
    dirty = subprocess.check_output(['git','-C',str(UPSTREAM),'status','--porcelain'],text=True).strip()
    if head != COMMIT or dirty:
        raise SystemExit('Reviewed NoRegrets checkout is missing or changed')
    build = ROOT / '.cache/noregrets-json'
    build.mkdir(exist_ok=True)
    mods = re.findall(r'^mod (\w+);', (UPSTREAM/'src/main.rs').read_text(), re.M)
    header = '\n'.join(f'#[allow(dead_code)] #[path={json.dumps(str(UPSTREAM / "src" / (m+".rs")))}] mod {m};' for m in mods)
    (build/'adapter.rs').write_text(header+'\n'+(ROOT/'native/noregrets-json/adapter.rs').read_text())
    (build/'bridge.rs').write_text(header+'\n'+(ROOT/'native/noregrets-json/bridge.rs').read_text())
    manifest = (UPSTREAM/'Cargo.toml').read_text()
    (build/'Cargo.toml').write_text(manifest+'\n[[bin]]\nname = "poker-json"\npath = "adapter.rs"\n\n[[bin]]\nname = "poker-bridge"\npath = "bridge.rs"\n')
    (build/'Cargo.lock').write_bytes((UPSTREAM/'Cargo.lock').read_bytes())
    (build/'LICENSE.upstream').write_bytes((UPSTREAM/'LICENSE').read_bytes())
    env = os.environ.copy()
    env.setdefault('CARGO_HOME','/workspace/.tools/cargo')
    env.setdefault('RUSTUP_HOME','/workspace/.tools/rustup')
    env['CARGO_TARGET_DIR'] = str(UPSTREAM/'target')
    env['CARGO_BUILD_JOBS'] = '4'
    subprocess.run([env['CARGO_HOME']+'/bin/cargo','+1.90.0','build','--release','--locked','--bin','poker-json','--bin','poker-bridge'], cwd=build,env=env,check=True)
    print(UPSTREAM/'target/release/poker-json')

if __name__ == '__main__': main()
