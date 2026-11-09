"""Build a separate rules oracle; preserve the original evaluator binaries."""
import json
import os
import subprocess
from build_json_adapter import ROOT,UPSTREAM,COMMIT


def main():
    assert subprocess.check_output(['git','-C',str(UPSTREAM),'rev-parse','HEAD'],text=True).strip()==COMMIT
    assert not subprocess.check_output(['git','-C',str(UPSTREAM),'status','--porcelain'],text=True).strip()
    dest=ROOT/'.cache/noregrets-rules';dest.mkdir(exist_ok=True)
    header='\n'.join(f'#[allow(dead_code)] #[path={json.dumps(str(UPSTREAM/"src"/(name+".rs")))}] mod {name};' for name in ('cards','eval','engine'))
    (dest/'rules.rs').write_text(header+'\n'+(ROOT/'native/noregrets-json/rules.rs').read_text())
    (dest/'Cargo.toml').write_text((UPSTREAM/'Cargo.toml').read_text()+'\n[[bin]]\nname="poker-rules"\npath="rules.rs"\n')
    (dest/'Cargo.lock').write_bytes((UPSTREAM/'Cargo.lock').read_bytes())
    (dest/'LICENSE.upstream').write_bytes((UPSTREAM/'LICENSE').read_bytes())
    env=dict(os.environ,CARGO_HOME='/workspace/.tools/cargo',RUSTUP_HOME='/workspace/.tools/rustup',CARGO_TARGET_DIR=str(UPSTREAM/'target'),CARGO_BUILD_JOBS='2')
    subprocess.run([env['CARGO_HOME']+'/bin/cargo','+1.90.0','build','--offline','--release','--locked','--bin','poker-rules'],cwd=dest,env=env,check=True)
    print(UPSTREAM/'target/release/poker-rules')


if __name__=='__main__':main()
