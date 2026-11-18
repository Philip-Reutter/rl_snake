"""
    Hyperparameter optimization in two stages (Optuna) only using CPU cores as model is small

    python scripts/hyperparameter_opt.py search  # stage 1: cheap, noisy screening
    Arguments:
    --study     # name of the Optuna study (default "snake")
    --jobs      # parallel training processes (one CPU core each)
    --episodes  # episodes per run (default 300)
    --trials    # number of new trials to add to the study

    python scripts/hyperparameter_opt.py confirm  # stage 2: the finalists vs. the baseline
    Arguments:
    --study     # name of the Optuna study (default "snake")
    --jobs      # parallel training processes (one CPU core each)
    --episodes  # episodes per run (default config.episodes)
    --top       # how many of the best trials to re-run (default 3)
    --seeds     # training seeds per candidate (default 3)

    Stage 1 trains short runs with one seed
    Optuna's TPE sampler proposes hyperparameters, and a median pruner stops runs that are clearly behind after their greedy evaluations
    The score of a trial is the mean food eaten by the final network on held-out games
    The first trial is current config, so the search has a baseline measured the same way

    Stage 2 trains best trials plus baseline again with new seeds for full number of episodes and compares them on the same 100 held-out games

    Results go to results/hpo/ (study.db, search_trials.csv, confirm.csv)
    The database is kept between runs, so `search` can be called again to add trials.
"""
import argparse
import csv
import multiprocessing
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import optuna

from src.config import RESULTS_DIR, Config 
from src.training import train

HPO_DIR = RESULTS_DIR / "hpo"
SEARCH_SEED = 0            # all search trials use the same training seed: differences come from the settings
CONFIRM_SEED_START = 1000  # confirmation seeds are never used in the search
SEARCH_EVAL_GAMES = 20     # greedy games per evaluation during search
SEARCH_FINAL_GAMES = 30    # held-out games for the score of a search trial
CONFIRM_FINAL_GAMES = 100  # held-out games per confirmation run

SPACE = {
    "lr": ("log", 3e-4, 2e-3),
    "gamma": ("float", 0.97, 0.995),
    #"hidden": ("cat", [64, 128, 256]),
    "batch_size": ("cat", [32, 64, 128]),
    "target_update_every": ("log", 100, 3000),
    "eps_decay": ("float", 0.98, 0.998),
    "eps_end": ("log", 0.005, 0.1),
    #"shaping": ("cat", [0.0, 0.01, 0.02, 0.05]),
}


def suggest(trial):
    params = {}
    for name, (kind, *rest) in SPACE.items():
        if kind == "cat":
            params[name] = trial.suggest_categorical(name, rest[0])
        elif name == "target_update_every":
            params[name] = trial.suggest_int(name, int(rest[0]), int(rest[1]), log=True)
        else:
            params[name] = trial.suggest_float(name, rest[0], rest[1], log=(kind == "log"))
    return params


def baseline_params(cfg):
    params = {}
    for name, (kind, *rest) in SPACE.items():
        value = getattr(cfg, name)
        params[name] = min(rest[0], key=lambda c: abs(c - value)) if kind == "cat" else value
    return params


def run_config(cfg, params, episodes, seed, eval_games):
    return replace(cfg, **params, episodes=episodes, seed=seed, eval_games=eval_games, milestones=())


def search_worker(study_name, storage, n_trials, worker_id, episodes):
    study = optuna.load_study(study_name=study_name, storage=storage,
                              sampler=optuna.samplers.TPESampler(seed=worker_id, multivariate=True),
                              pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=200))

    def objective(trial):
        cfg = run_config(Config(), suggest(trial), episodes, SEARCH_SEED, SEARCH_EVAL_GAMES)

        def on_eval(episode, mean_score):
            trial.report(mean_score, step=episode)
            if trial.should_prune():
                raise optuna.TrialPruned()

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            result = train(cfg, out / "ckpt", out / "res", on_eval=on_eval,
                           final_eval_games=SEARCH_FINAL_GAMES, plot=False, verbose=False)
        return float(np.mean(result["final_scores"]))

    # If worker fails to report a trial because database says it is finished, skip it
    for _ in range(n_trials):
        try:
            study.optimize(objective, n_trials=1)
        except ValueError as error:
            if "Cannot tell" not in str(error):
                raise
            print(f"worker {worker_id}: skipped one trial ({error})", flush=True)


def search(args):
    HPO_DIR.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{HPO_DIR / 'study.db'}"
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(study_name=args.study, storage=storage, direction="maximize",
                                load_if_exists=True)
    if not study.trials:
        study.enqueue_trial(baseline_params(Config()))
    done = len([t for t in study.trials if t.state.is_finished()])
    print(f"study '{args.study}': {done} finished trials, adding {args.trials} "
          f"({args.jobs} in parallel, {args.episodes} episodes each)", flush=True)

    counts = [args.trials // args.jobs + (i < args.trials % args.jobs) for i in range(args.jobs)]
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(args.jobs, mp_context=ctx) as pool:
        futures = [pool.submit(search_worker, args.study, storage, n, i + done, args.episodes)
                   for i, n in enumerate(counts) if n]
        for f in futures:
            f.result()

    study = optuna.load_study(study_name=args.study, storage=storage)
    write_trials(study)
    report_search(study)


def write_trials(study):
    with open(HPO_DIR / "search_trials.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["trial", "state", "score", *SPACE])
        for t in study.trials:
            writer.writerow([t.number, t.state.name, t.value, *(t.params.get(k) for k in SPACE)])


def report_search(study):
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials)
    print(f"\n{len(complete)} completed, {pruned} pruned. Top trials (held-out score of the final network):")
    for t in sorted(complete, key=lambda t: t.value, reverse=True)[:5]:
        print(f"  trial {t.number:3d}  score {t.value:5.1f}  {fmt(t.params)}")
    first = next((t for t in complete if t.number == 0), None)
    if first:
        print(f"  (trial 0 = current Config: {first.value:.1f})")
    print("\nThese scores come from one seed and are noisy. Next: python scripts/hyperparameter_opt.py confirm")


def fmt(params):
    return " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in params.items())


def confirm_run(cfg, params, episodes, seed):
    cfg = run_config(cfg, params, episodes, seed, Config().eval_games)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        result = train(cfg, out / "ckpt", out / "res", final_eval_games=CONFIRM_FINAL_GAMES,
                       plot=False, verbose=False)
    return float(np.mean(result["final_scores"]))


def confirm(args):
    storage = f"sqlite:///{HPO_DIR / 'study.db'}"
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.load_study(study_name=args.study, storage=storage)
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.number != 0]
    top = sorted(complete, key=lambda t: t.value, reverse=True)[:args.top]
    candidates = {"baseline (Config)": baseline_params(Config())}
    candidates.update({f"trial {t.number}": t.params for t in top})

    seeds = [CONFIRM_SEED_START + i for i in range(args.seeds)]
    print(f"{len(candidates)} candidates x {len(seeds)} seeds x {args.episodes} episodes, "
          f"{args.jobs} in parallel", flush=True)
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(args.jobs, mp_context=ctx) as pool:
        futures = {name: [pool.submit(confirm_run, Config(), params, args.episodes, s) for s in seeds]
                   for name, params in candidates.items()}
        results = {name: [f.result() for f in fs] for name, fs in futures.items()}

    HPO_DIR.mkdir(parents=True, exist_ok=True)
    with open(HPO_DIR / "confirm.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate", "mean", *[f"seed{s}" for s in seeds], *SPACE])
        for name, scores in results.items():
            writer.writerow([name, f"{np.mean(scores):.2f}", *[f"{s:.2f}" for s in scores],
                             *(candidates[name][k] for k in SPACE)])

    print(f"\nHeld-out score (mean of {CONFIRM_FINAL_GAMES} games, same games for everyone), one column per seed:")
    for name, scores in sorted(results.items(), key=lambda kv: -np.mean(kv[1])):
        per_seed = "  ".join(f"{s:6.1f}" for s in scores)
        print(f"  {name:18s} mean {np.mean(scores):6.1f} | {per_seed} | {fmt(candidates[name])}")
    best = max(results, key=lambda n: np.mean(results[n]))
    base = np.mean(results["baseline (Config)"])
    print(f"\nBest: {best}. Only adopt it if it beats the baseline ({base:.1f}) on most seeds")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["search", "confirm"])
    parser.add_argument("--study", default="snake", help="study name; use a new one after changing features/reward/space")
    parser.add_argument("--jobs", type=int, default=max(1, min(4, (os.cpu_count() or 2) - 1)),
                        help="parallel training processes (one CPU core each)")
    parser.add_argument("--trials", type=int, default=40, help="search: number of new trials")
    parser.add_argument("--episodes", type=int, help="episodes per run (search default 300, confirm default Config.episodes)")
    parser.add_argument("--top", type=int, default=3, help="confirm: how many of the best trials to re-run")
    parser.add_argument("--seeds", type=int, default=3, help="confirm: training seeds per candidate")
    args = parser.parse_args()

    if args.episodes is None:
        args.episodes = 300 if args.stage == "search" else Config().episodes
    (search if args.stage == "search" else confirm)(args)


if __name__ == "__main__":
    main()
