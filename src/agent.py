"""DQN agent: Q-network, Double-DQN update, target network, checkpoint helpers"""
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """obs -> one Q-value per action."""

    def __init__(self, obs_dim, n_actions, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


def greedy_action(net, obs, device="cpu"):
    with torch.no_grad():
        q = net(torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0))
    return int(q.argmax(dim=1).item())


class DQNAgent:
    def __init__(self, obs_dim, n_actions, cfg, device="cpu", seed=None):
        self.n_actions = n_actions
        self.cfg = cfg
        self.device = device
        self.online = QNetwork(obs_dim, n_actions, cfg.hidden).to(device)
        self.target = copy.deepcopy(self.online)  # frozen copy, refreshed periodically
        for p in self.target.parameters():
            p.requires_grad_(False)  # target network is not trained by gradient descent
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=cfg.lr)
        self.rng = np.random.default_rng(seed)
        self.learn_steps = 0

    def act(self, obs, epsilon):
        """Epsilon-greedy action."""
        if self.rng.random() < epsilon:
            return int(self.rng.integers(self.n_actions))
        return greedy_action(self.online, obs, self.device)

    def learn(self, buffer):
        """One gradient step on a replay batch. Returns the loss."""
        cfg = self.cfg
        obs, actions, rewards, next_obs, dones = (
            torch.as_tensor(x, device=self.device) for x in buffer.sample(cfg.batch_size))

        q = self.online(obs).gather(1, actions.unsqueeze(1)).squeeze(1) # Q(s, a) for the actions taken in the batch
        with torch.no_grad():
            # Double DQN: the online net picks the next action, the target net scores it.
            best = self.online(next_obs).argmax(dim=1, keepdim=True)  # argmax_a Q_online(s', a)
            q_next = self.target(next_obs).gather(1, best).squeeze(1)  # Q(s', argmax_a Q_online(s', a))
            y = rewards + cfg.gamma * (1.0 - dones) * q_next

        loss = F.smooth_l1_loss(q, y)  # Huber loss, less sensitive to outliers than MSE
        self.optimizer.zero_grad()
        loss.backward()  # compute gradients of loss
        nn.utils.clip_grad_norm_(self.online.parameters(), cfg.grad_clip)  # prevent exploding gradients
        self.optimizer.step()  # update online network parameters

        self.learn_steps += 1
        if self.learn_steps % cfg.target_update_every == 0:
            self.target.load_state_dict(self.online.state_dict())  # refresh target network with online network parameters
        return loss.item()


def save_checkpoint(path, net, obs_dim, n_actions, hidden, **meta):
    torch.save({"state_dict": net.state_dict(), "obs_dim": obs_dim,
                "n_actions": n_actions, "hidden": hidden, **meta}, path)


def load_checkpoint(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    net = QNetwork(ckpt["obs_dim"], ckpt["n_actions"], ckpt["hidden"]).to(device)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    meta = {k: v for k, v in ckpt.items() if k != "state_dict"}
    return net, meta
