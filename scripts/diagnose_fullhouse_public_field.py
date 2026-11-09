"""Run the diagnostic evaluator with five public neural opponents."""
import argparse
import hashlib
import json
from pathlib import Path

import diagnose_fullhouse_eval as diagnostic
from fullhouse_runtime import Bridge, Fullhouse


class PublicField(Bridge):
    def __init__(self, unused_registry_model=None):
        # Real rules engine only: no NoRegrets shadow/blueprint is needed.
        super().__init__()
        self.opponent = Fullhouse()
        self.state = None

    def call(self, request):
        if request.get('op') == 'blueprint':
            result = self.opponent.decide(self.state, request['seed'])
            return {'action': result['action'], 'fallback': False}
        self.state = super().call(request)
        return self.state


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--output',type=Path,required=True)
    args,_=parser.parse_known_args()
    diagnostic.evaluator.Bridge=PublicField
    diagnostic.main()
    path=args.output/'results.json'
    result=json.loads(path.read_text())
    result['field_runner_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result['opponent_field']='five public Fullhouse advantage policies'
    result['unused_registry_prerequisite_sha256']=result['model_hashes'].pop('noregrets')
    result['model_hashes']['opponent_public']=Fullhouse().digest
    result['decisions']={k.replace('noregrets/','opponent_public/'):v for k,v in result['decisions'].items()}
    result['latency']['opponent_public']=result['latency'].pop('noregrets')
    result['limitations'].append('Evaluator calls named blueprint were intercepted by PublicField; effective opponents and source hash recorded here.')
    text=json.dumps(result,indent=2)+'\n'
    path.write_text(text)
    (Path(result['raw_run'])/'results.json').write_text(text)


if __name__=='__main__':
    main()
