import numpy as np, sys
from dexmachina.rl.evaluate_dexmachina_spider import _spider_sr_errors, _align_obj_demo
# SPIDER success rate over ALL envs of a 256-env eval: success = pos_err<=0.1m AND quat_err<=0.5rad
p = sys.argv[1]
pos_thr = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1
quat_thr = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
d = np.load(p, allow_pickle=True).item()
obj = np.asarray(d["obj_state"]); demo = np.asarray(d["demo_state"])
if obj.ndim == 2: obj = obj[:, None, :]
if demo.ndim == 3: demo = demo[:, 0, :]
E = obj.shape[1]
n = 0; pes = []; qes = []
for e in range(E):
    o, dm = _align_obj_demo(obj[:, e, :], demo, "shift")
    pe, qe = _spider_sr_errors(o, dm)
    pes.append(pe); qes.append(qe)
    if pe <= pos_thr and qe <= quat_thr:
        n += 1
print(f"SR={n}/{E}={n/E:.4f} pos_err_med={np.median(pes):.4f} quat_err_med={np.median(qes):.4f}")
