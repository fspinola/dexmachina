import numpy as np, torch
from dexmachina.envs.demo_data import wrap_forearm_qpos_into_limits, _forearm_rotation, FOREARM_ROT_LIMIT as LIM

clips = [('ketchup_use_02', 's01', 40, 340, 'BAD  ketchup-40-340'),
         ('waffleiron_use_01', 's01', 40, 340, 'BAD  waffleiron-40-340'),
         ('box_use_01', 's01', 30, 230, 'GOOD box-30-230'),
         ('ketchup_use_01', 's01', 30, 130, 'GOOD ketchup-30-130'),
         ('mixer_use_01', 's01', 30, 200, 'GOOD mixer-30-200')]

def arr(v, fs, fe):
    a = v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v)
    return a[fs:fe].astype(float)

for base, sub, fs, fe, label in clips:
    for save in ('graph', 'para', 'ft'):
        import os
        p = f'dexmachina/assets/retargeted/allegro_hand/{sub}/{base}_vector_{save}.pt'
        if not os.path.exists(p):
            continue
        d = torch.load(p, weights_only=False, map_location='cpu')
        for side in ('L', 'R'):
            jq = d['retarget_data']['left' if side == 'L' else 'right']['joint_qpos']
            qd = {k: arr(v, fs, fe) for k, v in jq.items()}
            def g(ax, src):
                k = [k for k in src if f'{side}_forearm_{ax}' in k][0]
                return np.asarray(src[k], float)
            r0, p0, y0 = g('roll', qd), g('pitch', qd), g('yaw', qd)
            Rorig = _forearm_rotation(r0, p0, y0)
            out = wrap_forearm_qpos_into_limits(qd)
            r1, p1, y1 = g('roll', out), g('pitch', out), g('yaw', out)
            Rnew = _forearm_rotation(r1, p1, y1)
            inr = all(np.abs(x).max() <= LIM + 1e-6 for x in (r1, p1, y1))
            fk = np.degrees(max((Rnew[t].inv() * Rorig[t]).magnitude() for t in range(len(r1))))
            chg = not (np.allclose(r1, r0) and np.allclose(p1, p0) and np.allclose(y1, y0))
            maxjump = np.degrees(max(np.abs(np.diff(r1)).max(), np.abs(np.diff(y1)).max()))
            print(f'{label:22s} {save:5s} {side}: in_range={inr}  FKerr={fk:.4f}deg  changed={chg}  '
                  f'maxstep={maxjump:5.1f}deg  roll[{r1.min():6.2f},{r1.max():6.2f}] yaw[{y1.min():6.2f},{y1.max():6.2f}]')
