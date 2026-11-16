"""Strict NumPy-only export reader and inference for a window average."""
import numpy as np

FORMAT = 'compact-window-average-network-v1'

def load_average(path):
    with np.load(path,allow_pickle=False) as saved:
        fields={'format','metadata_json'}|{f'body.{i}.{k}' for i in (0,2,4) for k in ('weight','bias')}
        if set(saved.files)!=fields or str(saved['format'])!=FORMAT:
            raise ValueError('Incorrect average policy export')
        width=saved['body.0.weight'].shape[0]
        if not 1<=width<=256:
            raise ValueError('Invalid average width')
        shapes=((width,51),(width,width),(5,width));layers=[]
        for i,shape in zip((0,2,4),shapes):
            w,b=saved[f'body.{i}.weight'].copy(),saved[f'body.{i}.bias'].copy()
            if (w.shape!=shape or b.shape!=(shape[0],) or w.dtype!=np.float32 or b.dtype!=np.float32
                    or not np.isfinite(w).all() or not np.isfinite(b).all()):
                raise ValueError('Invalid average policy numeric layer')
            layers.append((w,b))
        import json
        metadata=json.loads(str(saved['metadata_json']))
    return tuple(layers),metadata


def average_strategy(layers,x,legal):
    x=np.asarray(x,dtype=np.float32);legal=np.asarray(legal,dtype=np.float32)
    if (x.shape!=(51,) or legal.shape!=(5,) or not np.isfinite(x).all()
            or not np.isin(legal,[0,1]).all() or not legal.any() or not np.array_equal(x[46:51],legal)):
        raise ValueError('Invalid average runtime observation')
    for index,(w,b) in enumerate(layers):
        x=np.dot(x,w.T)+b
        if index<2:x=np.maximum(x,0.)
    if not np.isfinite(x).all():raise FloatingPointError('Nonfinite average logits')
    p=np.exp(np.where(legal>0,x.astype(np.float64)-float(x[legal>0].max()),-np.inf))
    return (p/p.sum()).astype(np.float32)
