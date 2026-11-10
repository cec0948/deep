"""NumPy-only partition shared by training and lightweight deployment."""
import numpy as np


def is_opening(features):
    x=np.asarray(features)
    return (x[...,17]==1)&(x[...,24]==1)&np.isclose(x[...,27],.015)&(x[...,45]==1)


def base_strata(features):
    x=np.asarray(features)
    street=np.argmax(x[:,17:21],axis=1)
    pot=np.searchsorted(np.array([.1,.5,2.]),x[:,27],side='right')
    groups=6+4*street+pot
    unopened=(street==0)&(x[:,37]==0)&(x[:,41]*x[:,27]<=.01001)
    groups[unopened]=np.argmax(x[unopened,21:27],axis=1)
    return groups
