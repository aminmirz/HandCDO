"""Export the averaged in-hand rotational slip curves used in the experiment video, for the v2 page's live plots.

Uses HandGeneration/experiments/{visualize_trial,stability_analysis}.py unchanged (OptiTrack FBX + UR5e NPY per trial)
and repeats the post-processing of stability_analysis_animated.py exactly: average per hand, spoon hand_170 stretched
to the other hands' duration, spoon cropped to 5-20 s and shifted to 0-15 s, then hand_170 +0.8 s.

  python -B website_assets/scripts/export_slip_curves.py     # writes v2/media/experiments/slip.json
"""
import json, sys, types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = Path.home() / 'Desktop/PhD/_Projects/HandCoDesign/HandGeneration/experiments'
sys.path.insert(0, str(ROOT / '.runtime/python')); sys.path.insert(0, str(EXP))
# plotting is not needed; satisfy the modules' top-level matplotlib imports
mpl = types.ModuleType('matplotlib'); mpl.use = lambda *a, **k: None; mpl.rcParams = {}
for name in ('matplotlib', 'matplotlib.pyplot', 'matplotlib.animation'):
    sys.modules[name] = mpl
mpl.pyplot = mpl; mpl.animation = mpl; mpl.FuncAnimation = None
import numpy as np
import stability_analysis as sa

HANDS = {'hand_170': 'high', 'hand_13': 'mid', 'hand_1': 'low'}     # paper: iter 170 / 13 / 1 -> 0.81 / 0.60 / 0.45


def curves_for(tool):
    out = {}
    for hand in sa.HANDS:
        results = []
        folder = EXP / tool / hand
        for trial in range(len(sa.pair_files(folder))):
            if (hand, tool, trial) in sa.SKIP_TRIALS:
                continue
            r = sa.process_trial(hand, tool, trial)
            if r is not None:
                results.append(r[0])
        t, inst = sa.average_instability_curves(results)
        if t is not None:
            out[hand] = (t, inst)
    if tool == 'spoon' and 'hand_170' in out:                                       # same as the animation script
        target = max(out[h][0][-1] for h in out if h != 'hand_170')
        t, inst = out['hand_170']; out['hand_170'] = (t * (target / t[-1]), inst)
        for h in list(out):
            t, inst = out[h]; m = (t >= 5.0) & (t <= 20.0); out[h] = (t[m] - 5.0, inst[m])
        t, inst = out['hand_170']; out['hand_170'] = (t + 0.8, inst)
    return out


def main():
    data = {}
    for tool in ('spoon', 'hammer', 'knife'):
        c = curves_for(tool)
        max_dur = max(t[-1] for t, _ in c.values()); allv = np.concatenate([v for _, v in c.values()])
        hands = {}
        for hand, (t, v) in c.items():
            step = max(1, int(round(len(t) / (t[-1] - t[0] + 1e-9) / 60)))           # keep ~60 samples per second
            hands[HANDS[hand]] = dict(hand=hand, t=[round(float(x), 4) for x in t[::step]], v=[round(float(x), 3) for x in v[::step]],
                                      mean=round(float(np.mean(v)), 3))
        data[tool] = dict(x_max=round(float(max_dur), 4), y_lo=round(float(allv.min()) - .5, 3), y_hi=round(float(allv.max()) + .5, 3),
                          extra_delay=0.8 if tool == 'spoon' else 0.0, hands=hands)
        print(tool, {k: (len(h['t']), h['mean']) for k, h in hands.items()}, 'x_max', data[tool]['x_max'], flush=True)
    out = ROOT / 'v2/media/experiments'; out.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(dict(source='HandGeneration/experiments (OptiTrack + UR5e), stability_analysis.py', unit='deg', data=data))
    (out / 'slip.json').write_text(payload)
    (out / 'slip.js').write_text('window.HandCDOSlip = ' + payload + ';' + chr(10))        # loadable from file:// too


if __name__ == '__main__':
    main()
