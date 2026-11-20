from dataclasses import replace

import numpy as np
import torch

from src.agent import DQNAgent, load_checkpoint, save_checkpoint
from src.config import Config
from src.replay_buffer import ReplayBuffer

OBS_DIM, N_ACTIONS = 13, 3


def make_agent(**overrides):
    cfg = replace(Config(), hidden=32, batch_size=32, **overrides)
    torch.manual_seed(0)
    return DQNAgent(OBS_DIM, N_ACTIONS, cfg, seed=0), cfg


def toy_buffer(n=256):
    """Every transition ends episode, action 1 pays +1, others 0"""
    rng = np.random.default_rng(0)
    buf = ReplayBuffer(n, OBS_DIM, seed=0)
    for _ in range(n):
        a = int(rng.integers(N_ACTIONS))
        obs = rng.normal(size=OBS_DIM).astype(np.float32)
        buf.add(obs, a, 1.0 if a == 1 else 0.0, obs, True)
    return buf


def test_epsilon_extremes():
    agent, _ = make_agent()
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    assert {agent.act(obs, 1.0) for _ in range(200)} == {0, 1, 2}  # pure exploration
    assert len({agent.act(obs, 0.0) for _ in range(50)}) == 1  # greedy is deterministic


def test_learning_picks_the_rewarded_action():
    agent, _ = make_agent(lr=3e-3)
    buf = toy_buffer()
    first = agent.learn(buf)
    for _ in range(400):
        last = agent.learn(buf)
    assert last < first
    with torch.no_grad():
        q = agent.online(torch.as_tensor(buf.obs))
    assert (q.argmax(dim=1) == 1).float().mean() > 0.95  # network learned to pick rewarded action


def test_target_network_is_copied_only_periodically():
    agent, _ = make_agent(target_update_every=10)
    buf = toy_buffer()

    def same():
        return all(torch.equal(a, b) for a, b in zip(agent.online.parameters(), agent.target.parameters()))

    assert same()
    for _ in range(9):
        agent.learn(buf)
    assert not same()  # online net moved, target is frozen
    agent.learn(buf)
    assert same()  # 10th step -> copy


def test_checkpoint_roundtrip(tmp_path):
    agent, cfg = make_agent()
    path = tmp_path / "ck.pt"
    save_checkpoint(path, agent.online, OBS_DIM, N_ACTIONS, cfg.hidden, episode=7)
    net, meta = load_checkpoint(path)
    x = torch.randn(5, OBS_DIM)
    assert meta["episode"] == 7
    assert torch.allclose(net(x), agent.online(x))
