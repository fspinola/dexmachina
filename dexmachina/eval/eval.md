## Policy Evaluation and Result Reporting

In the paper we report results using an AUC-ADD metric that is designed to be a meaningful reflection of the articulated object tracking task performance.

- ADD (Average 3D Distance) metric: for each eval env and each part (`top`, `bottom`), transform a fixed set of sampled mesh vertices into world space for both `demo_state` and `obj_state`, compute per-vertex L2 distances, average over vertices to get per-frame distances, then store as `add.npy` with shape `(num_eval_envs, num_frames)` per part.
- AUC (Area Under the Curve): for each part, compute accuracy at thresholds `[0.01, 0.02, ..., 0.09]` as `mean(add < thres)`; then compute trapezoidal AUC over normalized x-values `linspace(0, 1, len(thresholds))`; final AUC is the mean over parts.

## Single policy to ADD-AUC stats (end-to-end)

1) Roll out a policy and produce `eval_ep0.npy` (use `--num_envs` to batch episodes):

```bash
python -m dexmachina.rl.eval_rl_games \
  --checkpoint /path/to/ckpt_dir/ckpt_name.pth \
  --num_envs 20
```

2) Compute ADD/AUC for that eval file (writes `add.npy` and `add_stats.json` next to it):

```bash
python -m dexmachina.eval.compute_add \
  --input /path/to/.../allegro_hand_eval/eval_ep0.npy
```

You can also scan a root directory:

```bash
STATS_PATH=logs/
python -m dexmachina.eval.compute_add \
  --input $STATS_PATH \
  --pattern "**/eval_ep*.npy"
```

## Group results for reporting

After running single-policy evaluations and saving per-policy eval data, results are averaged across multiple random seeds. Our original policy training was done across a large set of runs, so we group then using a `.yaml` config file which specify a few conditions that the code can use to filter and group across a long list of runs. See an example config in `eval/group_cfgs/`. But we have also provided a grouped list of all the runs below.

### Using the provided eval data

We have provided all the policy data used for our paper's main result table in [this link](https://drive.google.com/file/d/1d8sMncXvPir-PdiUFYW4t-J5YlEf8NFW/view?usp=sharing). This should include all runs' policy checkpoints, config files, pre-computed ADD results, etc. Once you unzip the data it should contain:

- `dexmachina_main_results_data/runs/*` for per-run data
- `dexmachina_main_results_data/run_paths.json` for the run list

All runs have been grouped by robot hand type and task, and the ADD-AUC metric has been computed, so you can use the provided `run_paths.json` to generate grouped stats:

```bash
cd $HOME/dexmachina/
DATA_ROOT=/path/to/dexmachina_main_results_data
python -m dexmachina.eval.group_results  --run_paths_json $DATA_ROOT/run_paths.json   --run_paths_base $DATA_ROOT   --pattern "**/eval_ep*.npy"   --use_auc   --output_name dexmachina_main_stats
```
This will output to `dexmachina/dexmachina/eval/stats`.


### Using your own data
For result reporting on your own data, you would need to add a new grouping config file in `eval/group_cfgs/` (or do your manual grouping in some other way and provide it as a `.json` run list like above) and report averaged ADD-AUC results, doing something like this:

```bash
cd $HOME/dexmachina/
STATS_PATH= # set path to runs here
python -m dexmachina.eval.group_results \
  --group_config YOUR_CFG \
  --config_path dexmachina/eval/group_cfgs \
  --input $STATS_PATH \
  --pattern "**/eval_ep*.npy" \
  --use_auc --output_name grouped_ADD_stats
```

The script loads `add_stats.json` (or falls back to `add.npy`) for each run, groups results using the config file, and writes a JSON summary under `dexmachina/eval/stats`.


