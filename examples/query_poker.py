"""Run from the repository root: python examples/query_poker.py MODEL.bin"""
import json
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
binary = root / '.cache/noregrets/target/release/poker-json'
if len(sys.argv) != 2:
    raise SystemExit('usage: python examples/query_poker.py MODEL.bin')
request = dict(hero=3, button=0, hole=['As','Kd'], board=[], actions=[], seed=42)
with subprocess.Popen([str(binary), sys.argv[1]], stdin=subprocess.PIPE,
                      stdout=subprocess.PIPE, text=True) as engine:
    # Keep this process alive for subsequent decisions; the model loads once.
    engine.stdin.write(json.dumps(request) + '\n')
    engine.stdin.flush()
    response = engine.stdout.readline()
    if not response:
        raise SystemExit('Engine exited before producing a response; inspect stderr')
    print(json.dumps(json.loads(response), ensure_ascii=False, indent=2))
    engine.stdin.close()
    engine.wait(timeout=10)
    if engine.returncode:
        raise SystemExit(engine.returncode)
