import numpy as np

from src.replay_buffer import ReplayBuffer


def test_add_and_sample_shapes():
    buf = ReplayBuffer(10, obs_dim=4, seed=0)
    for i in range(6):
        buf.add(np.full(4, i), i % 3, float(i), np.full(4, i + 1), i == 5)
    assert len(buf) == 6
    obs, actions, rewards, next_obs, dones = buf.sample(32)
    assert obs.shape == (32, 4) and next_obs.shape == (32, 4)
    assert actions.shape == rewards.shape == dones.shape == (32,)
    # transitions stay consistent: next_obs = obs + 1, reward = obs value
    assert np.all(next_obs[:, 0] == obs[:, 0] + 1)
    assert np.all(rewards == obs[:, 0])


def test_circular_overwrite_drops_oldest():
    buf = ReplayBuffer(5, obs_dim=2, seed=0)
    for i in range(8):
        buf.add(np.zeros(2), 0, float(i), np.zeros(2), False)
    assert len(buf) == 5
    assert set(buf.rewards.tolist()) == {3.0, 4.0, 5.0, 6.0, 7.0}
