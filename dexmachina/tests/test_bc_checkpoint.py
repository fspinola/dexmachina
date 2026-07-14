"""BC checkpoint <-> rl_games actor compatibility + tiny overfit test.

Run from the repo root:
    python -m pytest dexmachina/tests/test_bc_checkpoint.py -q
"""

import glob
import os

import pytest
import torch

from dexmachina.rl.train_bc_kinref import build_rl_games_model, save_checkpoint

OBS_DIM, ACTION_DIM = 341, 44
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EXPECTED_MODEL_KEYS = {
    "value_mean_std.running_mean", "value_mean_std.running_var", "value_mean_std.count",
    "a2c_network.sigma",
    "a2c_network.actor_mlp.0.weight", "a2c_network.actor_mlp.0.bias",
    "a2c_network.actor_mlp.2.weight", "a2c_network.actor_mlp.2.bias",
    "a2c_network.actor_mlp.4.weight", "a2c_network.actor_mlp.4.bias",
    "a2c_network.actor_mlp.6.weight", "a2c_network.actor_mlp.6.bias",
    "a2c_network.value.weight", "a2c_network.value.bias",
    "a2c_network.mu.weight", "a2c_network.mu.bias",
}


def test_model_matches_rl_games_checkpoint_schema():
    model = build_rl_games_model(OBS_DIM, ACTION_DIM, "cpu")
    keys = set(model.state_dict().keys())
    assert keys == EXPECTED_MODEL_KEYS
    sd = model.state_dict()
    assert tuple(sd["a2c_network.actor_mlp.0.weight"].shape) == (512, OBS_DIM)
    assert tuple(sd["a2c_network.mu.weight"].shape) == (ACTION_DIM, 128)
    assert tuple(sd["a2c_network.sigma"].shape) == (ACTION_DIM,)
    assert tuple(sd["a2c_network.value.weight"].shape) == (1, 128)


def test_checkpoint_round_trip_strict(tmp_path):
    model = build_rl_games_model(OBS_DIM, ACTION_DIM, "cpu")
    with torch.no_grad():
        model.a2c_network.sigma.fill_(-1.6)
    path = str(tmp_path / "bc_test.pth")
    save_checkpoint(path, model, {"marker": "test"})

    # The exact load path used by train_rl_games.py --warmstart_ckpt:
    # torch_ext.load_checkpoint -> agent.set_weights -> strict load_state_dict.
    from rl_games.algos_torch import torch_ext

    ckpt = torch_ext.load_checkpoint(path)
    assert ckpt["bc_metadata"]["marker"] == "test"
    fresh = build_rl_games_model(OBS_DIM, ACTION_DIM, "cpu")
    fresh.load_state_dict(ckpt["model"], strict=True)
    assert float(fresh.a2c_network.sigma.detach()[0]) == pytest.approx(-1.6)


@pytest.mark.skipif(
    not glob.glob(os.path.join(REPO, "logs/rl_games/*/*/nn/*.pth")),
    reason="no real RL checkpoint on disk",
)
def test_key_set_matches_a_real_rl_checkpoint():
    real_path = sorted(glob.glob(os.path.join(REPO, "logs/rl_games/*/*/nn/*.pth")))[0]
    real = torch.load(real_path, map_location="cpu", weights_only=False)
    model = build_rl_games_model(OBS_DIM, ACTION_DIM, "cpu")
    assert set(model.state_dict().keys()) == set(real["model"].keys()), (
        f"BC model keys diverge from real RL checkpoint {real_path}"
    )


def test_tiny_overfit():
    """MSE on a tiny synthetic set must drop by >=50x — catches broken labels,
    wrong head, or optimizer wiring."""
    torch.manual_seed(0)
    model = build_rl_games_model(OBS_DIM, ACTION_DIM, "cpu")
    obs = torch.randn(128, OBS_DIM)
    target = torch.tanh(obs[:, :ACTION_DIM] * 0.3)  # smooth deterministic map
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    sigma_before = model.a2c_network.sigma.clone()
    value_w_before = model.a2c_network.value.weight.clone()

    def loss_now():
        mu, _, _, _ = model.a2c_network({"obs": obs})
        return torch.nn.functional.mse_loss(mu, target)

    initial = float(loss_now())
    for _ in range(300):
        loss = loss_now()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    final = float(loss_now())
    assert final < initial / 50, f"no overfit: {initial:.4f} -> {final:.4f}"
    # BC must not touch the fixed log-std or the value head weights.
    assert torch.equal(model.a2c_network.sigma, sigma_before)
    assert torch.equal(model.a2c_network.value.weight, value_w_before)
