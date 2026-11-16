"""Inference for a bounded continuation-window average; no Torch/history lookup."""
import json
from pathlib import Path

import numpy as np

from fullhouse_runtime import Fullhouse
from fullhouse_v4_runtime import FullhouseV4
from fullhouse_v3_common import digest
from poker_average_inference import FORMAT,average_strategy,load_average
from rich_poker_runtime import policy_for as existing_policy_for


class WindowAveragePolicy(FullhouseV4):
    def __init__(self,model,source):
        # Fullhouse supplies action translation, immutable equity tables and the
        # public-state feature interface. Its default opponent weights are not
        # consulted by this policy's overridden get_strategy().
        Fullhouse.__init__(self)
        import importlib.util
        source=Path(source).resolve()
        spec=importlib.util.spec_from_file_location('window_average_features',source/'bot/features.py')
        self.features=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.features)
        self.model=Path(model).resolve();self.digest=digest(self.model)
        self.layers,self.metadata=load_average(self.model)
        self.model_type=FORMAT;self.lookup.get_strategy=self._strategy
        self.dependency_hashes={str(self.model):self.digest,
            str(source/'bot/features.py'):digest(source/'bot/features.py')}
        for name in ('poker_average_runtime.py','poker_average_inference.py'):
            path=Path(__file__).with_name(name).resolve();self.dependency_hashes[str(path)]=digest(path)

    def _strategy(self,state):
        x,legal=self.features.encode_state_dict(state,self.lookup._equity_tables)
        return average_strategy(self.layers,x,legal)

    def decide(self,state,seed):
        result=Fullhouse.decide(self,state,seed)
        result.update(policy_source='continuation_window_average',
            averaging_window=self.metadata.get('averaging_window'),
            historical_mixture_available=False,strength_verified=False)
        return result


def policy_for(model,source=None):
    path=Path(model)
    if path.suffix=='.npz':
        with np.load(path,allow_pickle=False) as data:
            if 'format' in data and str(data['format'])==FORMAT:
                if source is None:raise ValueError('Frozen source required for average inference')
                return WindowAveragePolicy(path,source)
    return existing_policy_for(path,source)
