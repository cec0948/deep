"""Serve an explicitly selected flop repair using the usual poker JSONL protocol."""
import argparse
from contextlib import closing
import json
from pathlib import Path
import sys

from fullhouse_runtime import Bridge, validate_request
from poker_flop_repair import FlopRepairPolicy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('policy', type=Path)
    args = parser.parse_args()
    policy = FlopRepairPolicy(args.policy)
    with closing(Bridge()) as engine:
        for line in sys.stdin:
            try:
                if len(line) > 1048576:
                    raise ValueError('Request exceeds 1 MiB')
                request = json.loads(line)
                validate_request(request)
                state = engine.call(dict(request, op='replay'))
                reply = policy.decide(state, request.get('seed', 0))
                reply.update(backend='flop-repair', policy_manifest_sha256=policy.digest)
            except (ValueError, TypeError, KeyError, FloatingPointError) as err:
                reply = dict(ok=False, error=str(err))
            print(json.dumps(reply, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
