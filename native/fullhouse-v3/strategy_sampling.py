# Appended to the deterministic R1 support module by build_fullhouse_v3.py.
REVISION = 'r1-cpu-v3'
STRATEGY_OPTIMIZER = 'feature-stratified-pps-swa-v1'
STRATEGY_BATCH_SIZE = 512


def strategy_strata(features):
    """Feature-defined strata: six unopened positions, then street x pot size.

    Multiplying all weights in a stratum by the same constant preserves the
    conditional average at every identical feature vector. It changes the
    projection distribution for a finite shared network, which is intentional.
    """
    x=np.asarray(features)
    street=np.argmax(x[:,17:21],axis=1)
    pot=np.searchsorted(np.array([.1,.5,2.]),x[:,27],side='right')
    groups=6+4*street+pot
    unopened=(street==0)&(x[:,37]==0)&(x[:,41]*x[:,27]<=.01001)
    groups[unopened]=np.argmax(x[unopened,21:27],axis=1)
    return groups


class StrategySampler:
    """Uniform positive-mass stratum, PPS within stratum, unweighted CE."""
    def __init__(self,buffer):
        self.buffer=buffer;n=buffer.size
        if n<1:raise ValueError('Empty strategy replay')
        self.groups=strategy_strata(buffer.features[:n])
        w=time_weights(buffer.iterations[:n].astype(np.float64))*buffer.sample_weights[:n]
        if not np.isfinite(w).all() or (w<0).any() or w.max()<=0:
            raise ValueError('Invalid strategy weights')
        scaled=w/w.max()
        totals=np.bincount(self.groups,weights=scaled,minlength=22)
        count=np.count_nonzero(totals)
        self.probabilities=np.divide(scaled,totals[self.groups],out=np.zeros_like(scaled),where=totals[self.groups]>0)/count
        self.cdf=np.cumsum(self.probabilities);self.cdf[-1]=1.
        self.diagnostics={str(k):dict(rows=int((self.groups==k).sum()),
            original_mass=float(totals[k]/totals.sum()),
            **weight_diagnostics(w[self.groups==k])) for k in np.flatnonzero(totals)}

    def sample(self,batch_size):
        ids=np.searchsorted(self.cdf,np.random.random(batch_size),side='right')
        b=self.buffer
        return b.features[ids],b.advantages[ids],np.ones(batch_size,dtype=np.float32)


def write_strategy_audit(path,iteration,net,buffer):
    """Root conditional averages are observable exact targets in finite replay."""
    n=buffer.size;x=buffer.features[:n];y=buffer.advantages[:n]
    w=time_weights(buffer.iterations[:n].astype(float))*buffer.sample_weights[:n]
    root=(x[:,17]==1)&(x[:,24]==1)&np.isclose(x[:,27],.015)
    ids=np.flatnonzero(root);rows=[]
    if len(ids):
        # All features must agree, not just a hand-strength bucket.
        keys,inverse=np.unique(x[ids],axis=0,return_inverse=True)
        mass=np.bincount(inverse,weights=w[ids],minlength=len(keys))
        count=np.bincount(inverse,minlength=len(keys))
        targets=np.stack([np.bincount(inverse,weights=w[ids]*y[ids,a],minlength=len(keys)) for a in range(5)],1)/mass[:,None]
        with torch.no_grad():
            t=torch.from_numpy(keys)
            pred=net.get_strategy(t,t[:,46:51]).numpy()
        tv=np.abs(pred-targets).sum(1)/2
        for i in range(len(keys)):
            if keys[i,10]==1 and keys[i,7]>=11/12-1e-6:
                rows.append(dict(hand='AA' if keys[i,7]>.99 else 'KK',rows=int(count[i]),target=targets[i].tolist(),prediction=pred[i].tolist(),tv=float(tv[i])))
        root_score=dict(rows=len(ids),classes=len(keys),weighted_tv=float(np.average(tv,weights=mass)),premium_pairs=rows)
    else:root_score={'rows':0}
    row=dict(iteration=iteration,objective=STRATEGY_OPTIMIZER,root=root_score,strata=StrategySampler(buffer).diagnostics)
    with open(path,'a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
