"""Scalar prediction streams for the task experiments.

Each task pre-generates, per seed, a training series and a disjoint evaluation series (input ``u_t``,
target ``y_t``), standardised with training statistics.

``delay:<tau>``   ``y_t = u_{t-tau}``, ``u`` white uniform                       (linear memory)
``narma10``       the NARMA-10 system driven by ``u ~ U[0, 0.5]``                (nonlinear memory)
``mackey``        Mackey-Glass (``tau = 17``), predict ``x_{t+1}`` from ``x_t``
``lorenz``        Lorenz x-coordinate, predict the next sample from the current one
``sunspot``       monthly mean sunspot number (SILSO v2.0), next-step prediction; the first 80% of the
                  record is the training series and the last 20% the evaluation series
``power``         household electric power (UCI, 10-minute means), next-step prediction; seed ``s`` uses
                  its own 8000-bin segment of the record (see ``data/prepare_household_power.py``), the
                  first 80% for training and the last 20% for evaluation, so seeds share no observation
"""

from __future__ import annotations

import functools

import numpy as np
import torch

from .config import DTYPE
from .paths import DATA_DIR

TASKS = ("delay:5", "narma10", "mackey", "lorenz", "sunspot")
POWER = "power"  # a real series long enough for disjoint data per seed; not among the default tasks


def task_tag(task: str) -> str:
    """File-name form of a task name (``delay:5`` -> ``delay5``)."""
    return task.replace(":", "")


def _narma10(u):
    y = np.zeros_like(u)
    for t in range(10, len(u) - 1):
        y[t + 1] = 0.3 * y[t] + 0.05 * y[t] * y[t - 9 : t + 1].sum() + 1.5 * u[t - 9] * u[t] + 0.1
    return y


def _mackey_glass(length, rng, tau=17):
    history = list(1.2 + 0.2 * rng.random(tau + 1))
    x = history[-1]
    out = []
    for _ in range(length + 500):
        delayed = history[-tau]
        x = x + (0.2 * delayed / (1 + delayed**10) - 0.1 * x)
        history.append(x)
        out.append(x)
    return np.array(out[500:])


def _lorenz(length, rng, dt=0.01, subsample=2):
    x = np.array([1.0, 1.0, 1.0]) + 0.1 * rng.standard_normal(3)
    out = []
    for i in range(length * subsample + 1000):
        dx = np.array([10 * (x[1] - x[0]), x[0] * (28 - x[2]) - x[1], x[0] * x[1] - 8 / 3 * x[2]])
        x = x + dt * dx
        if i >= 1000 and (i - 1000) % subsample == 0:
            out.append(x[0])
    return np.array(out[:length])


@functools.lru_cache(maxsize=1)
def _power_segments():
    return np.loadtxt(DATA_DIR / "household_power_segments.txt")


def make_series(task, seed, train_length, eval_length):
    """Returns ``(u_train, y_train, u_eval, y_eval)`` as one-dimensional float arrays, standardised."""
    rng = np.random.default_rng(seed)
    length = train_length + eval_length + 200
    if task.startswith("delay:"):
        tau = int(task.split(":")[1])
        u = rng.uniform(-1, 1, length)
        y = np.roll(u, tau)
        u, y = u[100:], y[100:]
    elif task == "narma10":
        while True:  # NARMA-10 occasionally diverges; redraw the input until bounded
            u = rng.uniform(0, 0.5, length)
            y = _narma10(u)
            if np.isfinite(y).all() and np.abs(y).max() < 5:
                break
        u, y = u[100:], y[100:]
    elif task == "mackey":
        s = _mackey_glass(length, rng)
        u, y = s[:-1], s[1:]
    elif task == "lorenz":
        s = _lorenz(length, rng)
        u, y = s[:-1], s[1:]
    elif task == "sunspot":
        s = np.array([float(v) for v in open(DATA_DIR / "sunspot.txt").read().split()])
        u, y = s[:-1], s[1:]
        cut = int(0.8 * len(u))
        mu, sd = u[:cut].mean(), u[:cut].std()
        return (u[:cut] - mu) / sd, (y[:cut] - mu) / sd, (u[cut:] - mu) / sd, (y[cut:] - mu) / sd
    elif task == POWER:
        segments = _power_segments()
        if not 0 <= seed < len(segments):
            raise ValueError(f"task {POWER!r} has {len(segments)} disjoint segments; seed {seed} has none")
        s = segments[seed]
        u, y = s[:-1], s[1:]
        cut = int(0.8 * len(u))
        mu, sd = u[:cut].mean(), u[:cut].std()
        return (u[:cut] - mu) / sd, (y[:cut] - mu) / sd, (u[cut:] - mu) / sd, (y[cut:] - mu) / sd
    else:
        raise ValueError(task)
    u_train, y_train = u[:train_length], y[:train_length]
    u_eval, y_eval = u[train_length : train_length + eval_length], y[train_length : train_length + eval_length]
    mu_u, sd_u, mu_y, sd_y = u_train.mean(), u_train.std(), y_train.mean(), y_train.std()
    return (u_train - mu_u) / sd_u, (y_train - mu_y) / sd_y, (u_eval - mu_u) / sd_u, (y_eval - mu_y) / sd_y


class PredictionStream:
    """One series per seed, each replicated ``replicas`` times (replicas share the data).

    :meth:`step` cycles through the training series and reports wrap-arounds so that the caller can reset
    the state; :meth:`eval_iter` walks the evaluation series once.
    """

    def __init__(self, task, seeds, replicas, device, train_length=50_000, eval_length=10_000):
        series = [make_series(task, seed, train_length, eval_length) for seed in seeds]

        def to_lanes(i):
            stacked = np.stack([x[i] for x in series])
            return torch.tensor(stacked, dtype=DTYPE, device=device).repeat_interleave(replicas, 0)

        self.u_train, self.y_train, self.u_eval, self.y_eval = to_lanes(0), to_lanes(1), to_lanes(2), to_lanes(3)
        self.B, self.t = self.u_train.shape[0], 0
        self.train_length, self.eval_length = self.u_train.shape[1], self.u_eval.shape[1]

    def step(self):
        """Next training sample ``(x, y, wrapped)``; ``wrapped`` is true at the start of every pass."""
        wrapped = self.t == 0
        x, y = self.u_train[:, self.t : self.t + 1], self.y_train[:, self.t : self.t + 1]
        self.t = (self.t + 1) % self.train_length
        return x, y, wrapped

    def eval_iter(self):
        for t in range(self.eval_length):
            yield self.u_eval[:, t : t + 1], self.y_eval[:, t : t + 1]
