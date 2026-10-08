"""Convert MANO_RIGHT.pkl (chumpy pickle) to a plain-numpy mano_right.npz for mano_hand.py, without chumpy.

Only the mean-shape fields are kept, so the chumpy-wrapped shapedirs are not needed:
  python convert_mano.py data/mano/models/MANO_RIGHT.pkl data/mano/models/mano_right.npz
"""
import pickle
import sys

import numpy as np


class _Stub:
    def __init__(self, *a, **k): pass
    def __setstate__(self, state): self.__dict__.update(state if isinstance(state, dict) else {'_state': state})


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith(('chumpy', 'scipy')): return type(f'{module}.{name}', (_Stub,), {})
        return super().find_class(module, name)


def main(src, dst):
    d = _Unpickler(open(src, 'rb'), encoding='latin1').load()
    jr = d['J_regressor'].__dict__; rows, cols = jr['_shape']; J = np.zeros((rows, cols))   # scipy csc matrix, rebuilt dense
    for c in range(cols):
        for p in range(jr['indptr'][c], jr['indptr'][c + 1]): J[jr['indices'][p], c] = jr['data'][p]
    np.savez_compressed(dst, v_template=d['v_template'], f=d['f'].astype(np.int32), J_regressor=J, kintree=d['kintree_table'],
                        weights=d['weights'], posedirs=d['posedirs'], hands_mean=d['hands_mean'], hands_components=d['hands_components'])
    print('saved', dst)


if __name__ == '__main__':
    main(*sys.argv[1:3])
